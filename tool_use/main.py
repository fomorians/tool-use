import os
import attr
import trfl
import random
import argparse
import numpy as np
import tensorflow as tf
import pyoneer.rl as pyrl

from tqdm import trange
from gym.wrappers import TimeLimit

from tool_use.env import KukaEnv
from tool_use.models import Policy, Value
from tool_use.rollout import Rollout


@attr.s
class HyperParams:
    learning_rate = attr.ib(default=1e-3)
    grad_clipping = attr.ib(default=10.0)
    batch_size = attr.ib(default=10)
    iters = attr.ib(default=10)
    epochs = attr.ib(default=10)
    episodes = attr.ib(default=100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', required=True)
    parser.add_argument('--seed', default=42)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    print(args)

    tf.enable_eager_execution()

    env = KukaEnv(render=args.render)
    env = TimeLimit(env, max_episode_steps=1000)

    env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    summary_writer = tf.contrib.summary.create_file_writer(
        args.job_dir, max_queue=1, flush_millis=1000)
    summary_writer.set_as_default()

    params = HyperParams()
    print(params)

    rollout = Rollout(env, max_episode_steps=1000)

    action_size = env.action_space.shape[0]

    behavioral_policy = Policy(action_size=action_size)
    policy = Policy(action_size=action_size)
    value = Value()

    exploration_strategy = pyrl.strategies.SampleStrategy(behavioral_policy)
    inference_strategy = pyrl.strategies.ModeStrategy(behavioral_policy)

    global_step = tf.train.create_global_step()
    optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)

    checkpoint = tf.train.Checkpoint(
        global_step=global_step,
        optimizer=optimizer,
        behavioral_policy=behavioral_policy,
        policy=policy,
        value=value)
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
        states, actions, rewards, next_states, weights = rollout(
            exploration_strategy, episodes=params.episodes)
        if params.batch_size is not None:
            dataset = tf.data.Dataset.from_tensor_slices(
                (states, actions, rewards, next_states, weights))
            dataset = dataset.batch(params.batch_size)
        else:
            dataset = tf.data.Dataset.from_tensors((states, actions, rewards,
                                                    next_states, weights))
        for states, actions, rewards, next_states, weights in dataset:
            for epoch in range(params.epochs):
                grads_and_vars = agent.estimate_gradients(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    weights=weights,
                    global_step=global_step)
                grads, _ = zip(*grads_and_vars)
                grads_clipped, grads_norm = tf.clip_by_global_norm(
                    grads, params.grad_clipping)
                grads_clipped_norm = tf.global_norm(grads_clipped)
                grads_and_vars = zip(grads_clipped, agent.trainable_variables)
                optimizer.apply_gradients(
                    grads_and_vars, global_step=global_step)

                episodic_reward = tf.reduce_mean(
                    tf.reduce_sum(rewards, axis=-1))

                with tf.contrib.summary.always_record_summaries():
                    tf.contrib.summary.scalar('losses/policy_gradient',
                                              agent.policy_gradient_loss)
                    tf.contrib.summary.scalar(
                        'losses/entropy', agent.policy_gradient_entropy_loss)
                    tf.contrib.summary.scalar('losses/value', agent.value_loss)
                    tf.contrib.summary.scalar('grads_norm', grads_norm)
                    tf.contrib.summary.scalar('grads_norm/clipped',
                                              grads_clipped_norm)

            with tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar('rewards/train', episodic_reward)

        trfl.update_target_variables(
            source_variables=policy.trainable_variables,
            target_variables=behavioral_policy.trainable_variables)

        states, actions, rewards, next_states, weights = rollout(
            inference_strategy, episodes=params.episodes)
        episodic_reward = tf.reduce_mean(tf.reduce_sum(rewards, axis=-1))
        with tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar('rewards/eval', episodic_reward)

        checkpoint_prefix = os.path.join(args.job_dir, 'ckpt')
        checkpoint.save(file_prefix=checkpoint_prefix)


if __name__ == '__main__':
    main()
