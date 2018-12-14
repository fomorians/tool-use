import os
import trfl
import random
import argparse
import numpy as np
import tensorflow as tf

import pyoneer.rl as pyrl

from tqdm import trange
from gym.envs.classic_control import PendulumEnv
from gym.wrappers import TimeLimit

from tool_use.env import KukaEnv
from tool_use.models import Policy, Value, StateModel
from tool_use.params import HyperParams
from tool_use.rollout import Rollout
from tool_use.normalizer import Normalizer


def create_dataset(tensors, batch_size=None, shuffle=False, buffer_size=10000):
    if batch_size is not None:
        dataset = tf.data.Dataset.from_tensor_slices(tensors)
        if shuffle:
            dataset = dataset.shuffle(buffer_size)
        dataset = dataset.batch(batch_size)
    else:
        dataset = tf.data.Dataset.from_tensors(tensors)
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', required=True)
    parser.add_argument('--seed', default=42)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    print(args)

    params = HyperParams()
    print(params)

    tf.enable_eager_execution()

    env = PendulumEnv()
    # env = KukaEnv(render=args.render)
    env = TimeLimit(env, max_episode_steps=params.max_episode_steps)

    env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    summary_writer = tf.contrib.summary.create_file_writer(
        args.job_dir, max_queue=1, flush_millis=1000)
    summary_writer.set_as_default()

    rollout = Rollout(env, max_episode_steps=params.max_episode_steps)

    action_size = env.action_space.shape[0]

    behavioral_policy = Policy(action_size=action_size)
    policy = Policy(action_size=action_size)
    value = Value()
    state_model = StateModel()
    state_model_rand = StateModel()

    exploration_strategy = pyrl.strategies.SampleStrategy(behavioral_policy)
    inference_strategy = pyrl.strategies.ModeStrategy(behavioral_policy)

    global_step = tf.train.create_global_step()
    optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
    state_optimizer = tf.train.AdamOptimizer(
        learning_rate=params.learning_rate)

    extrinsic_reward_normalizer = Normalizer(
        center=False, scale=False, scale_by_max=True)
    intrinsic_reward_normalizer = Normalizer(
        center=False, scale=False, scale_by_max=True)

    checkpoint = tf.train.Checkpoint(
        global_step=global_step,
        optimizer=optimizer,
        state_optimizer=state_optimizer,
        behavioral_policy=behavioral_policy,
        policy=policy,
        value=value,
        state_model=state_model,
        state_model_rand=state_model_rand,
        extrinsic_reward_normalizer=extrinsic_reward_normalizer,
        intrinsic_reward_normalizer=intrinsic_reward_normalizer)
    checkpoint_path = tf.train.latest_checkpoint(args.job_dir)
    if checkpoint_path is not None:
        checkpoint.restore(checkpoint_path)

    state_size = env.observation_space.shape[0]
    mock_states = tf.zeros(shape=(1, 1, state_size), dtype=np.float32)
    behavioral_policy(mock_states)
    policy(mock_states)
    value(mock_states)

    trfl.update_target_variables(
        source_variables=policy.variables,
        target_variables=behavioral_policy.variables)

    agent = pyrl.agents.ProximalPolicyOptimizationAgent(
        policy=policy,
        behavioral_policy=behavioral_policy,
        value=value,
        optimizer=optimizer)

    for it in trange(params.iters):
        transitions = rollout(
            exploration_strategy, episodes=params.episodes, render=args.render)

        dataset = create_dataset(
            transitions, batch_size=params.batch_size, shuffle=True)

        for (states, actions, extrinsic_rewards, next_states,
             weights) in dataset:
            state_model_rand_pred = state_model_rand(states)

            with tf.GradientTape() as tape:
                state_model_pred = state_model(states)
                loss = tf.losses.mean_squared_error(
                    predictions=state_model_pred,
                    labels=tf.stop_gradient(state_model_rand_pred),
                    weights=weights[..., None])

            grads = tape.gradient(loss, state_model.trainable_variables)
            grads_clipped, grads_norm = tf.clip_by_global_norm(
                grads, params.grad_clipping)
            grads_clipped_norm = tf.global_norm(grads_clipped)
            grads_and_vars = zip(grads_clipped,
                                 state_model.trainable_variables)
            state_optimizer.apply_gradients(
                grads_and_vars, global_step=global_step)

            with tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar('losses/state_model', loss)
                tf.contrib.summary.scalar('grads_norm/state_model', grads_norm)
                tf.contrib.summary.scalar('grads_norm/state_model/clipped',
                                          grads_clipped_norm)

            intrinsic_rewards = tf.reduce_sum(
                tf.squared_difference(state_model_pred, state_model_rand_pred),
                axis=-1)

            extrinsic_rewards_norm = extrinsic_reward_normalizer(
                extrinsic_rewards, training=True) / params.max_episode_steps
            intrinsic_rewards_norm = intrinsic_reward_normalizer(
                intrinsic_rewards, training=True) / params.max_episode_steps
            rewards = tf.stop_gradient(extrinsic_rewards_norm +
                                       intrinsic_rewards_norm) / 2

            for epoch in range(params.epochs):
                grads_and_vars = agent.estimate_gradients(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    weights=weights,
                    global_step=global_step,
                    decay=params.decay,
                    lambda_=params.lambda_)
                grads, _ = zip(*grads_and_vars)
                grads_clipped, grads_norm = tf.clip_by_global_norm(
                    grads, params.grad_clipping)
                grads_clipped_norm = tf.global_norm(grads_clipped)
                grads_and_vars = zip(grads_clipped, agent.trainable_variables)
                optimizer.apply_gradients(
                    grads_and_vars, global_step=global_step)

                with tf.contrib.summary.always_record_summaries():
                    tf.contrib.summary.scalar('losses/policy_gradient',
                                              agent.policy_gradient_loss)
                    tf.contrib.summary.scalar(
                        'losses/entropy', agent.policy_gradient_entropy_loss)
                    tf.contrib.summary.scalar('losses/value', agent.value_loss)
                    tf.contrib.summary.scalar('grads_norm/agent', grads_norm)
                    tf.contrib.summary.scalar('grads_norm/agent/clipped',
                                              grads_clipped_norm)

            episodic_extrinsic_rewards = tf.reduce_mean(
                tf.reduce_sum(extrinsic_rewards_norm, axis=-1))
            episodic_intrinsic_rewards = tf.reduce_mean(
                tf.reduce_sum(intrinsic_rewards_norm, axis=-1))
            episodic_rewards = tf.reduce_mean(tf.reduce_sum(rewards, axis=-1))

            with tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar('extrinsic_rewards/train',
                                          episodic_extrinsic_rewards)
                tf.contrib.summary.scalar('intrinsic_rewards/train',
                                          episodic_intrinsic_rewards)
                tf.contrib.summary.scalar('rewards/train', episodic_rewards)

        trfl.update_target_variables(
            source_variables=policy.trainable_variables,
            target_variables=behavioral_policy.trainable_variables)

        states, actions, extrinsic_rewards, next_states, weights = rollout(
            inference_strategy, episodes=params.episodes, render=args.render)
        episodic_extrinsic_rewards = tf.reduce_mean(
            tf.reduce_sum(extrinsic_rewards, axis=-1))
        with tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar('extrinsic_rewards/eval',
                                      episodic_extrinsic_rewards)

        checkpoint_prefix = os.path.join(args.job_dir, 'ckpt')
        checkpoint.save(file_prefix=checkpoint_prefix)


if __name__ == '__main__':
    main()
