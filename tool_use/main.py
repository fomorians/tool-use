import os
import trfl
import random
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import pyoneer.rl as pyrl

from tqdm import trange
from gym.envs.classic_control import PendulumEnv

from tool_use import losses
from tool_use import targets
from tool_use.env import KukaEnv
from tool_use.models import Policy, Value
from tool_use.params import HyperParams
from tool_use.rollout import Rollout
from tool_use.normalizer import Normalizer


def create_dataset(tensors, batch_size=None, shuffle=True, buffer_size=10000):
    if batch_size is not None:
        dataset = tf.data.Dataset.from_tensor_slices(tensors)
        if shuffle:
            dataset = dataset.shuffle(buffer_size)
        dataset = dataset.batch(batch_size)
    else:
        dataset = tf.data.Dataset.from_tensors(tensors)
    return dataset


class PolicyWrapper:
    def __init__(self, policy, states_normalizer, actions_normalizer):
        self.policy = policy
        self.states_normalizer = states_normalizer
        self.actions_normalizer = actions_normalizer

    def __call__(self, state):
        state = tf.convert_to_tensor(state[None, None, ...], dtype=np.float32)
        state_norm = self.states_normalizer(state)
        action_norm = self.policy(state_norm)
        action = self.actions_normalizer.inverse(action_norm)
        action = action[0, 0].numpy()
        return action


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

    env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    rollout = Rollout(env, max_episode_steps=params.max_episode_steps)

    observation_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    behavioral_policy = Policy(action_size=action_size, scale=params.scale)
    policy = Policy(action_size=action_size, scale=params.scale)
    value = Value()

    global_step = tf.train.create_global_step()
    optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)

    states_normalizer = Normalizer(shape=[3], center=True, scale=True)
    actions_normalizer = Normalizer(shape=[1], center=False, scale=False)
    rewards_normalizer = Normalizer(center=False, scale=True)

    checkpoint = tf.train.Checkpoint(
        global_step=global_step,
        optimizer=optimizer,
        behavioral_policy=behavioral_policy,
        policy=policy,
        value=value,
        states_normalizer=states_normalizer,
        actions_normalizer=actions_normalizer,
        rewards_normalizer=rewards_normalizer)
    checkpoint_path = tf.train.latest_checkpoint(args.job_dir)
    if checkpoint_path is not None:
        checkpoint.restore(checkpoint_path)

    summary_writer = tf.contrib.summary.create_file_writer(
        args.job_dir, max_queue=1, flush_millis=1000)
    summary_writer.set_as_default()

    exploration_strategy = pyrl.strategies.SampleStrategy(behavioral_policy)
    inference_strategy = pyrl.strategies.ModeStrategy(behavioral_policy)

    exploration_wrapper = PolicyWrapper(exploration_strategy,
                                        states_normalizer, actions_normalizer)
    inference_wrapper = PolicyWrapper(inference_strategy, states_normalizer,
                                      actions_normalizer)

    # prime models
    mock_states = tf.zeros(shape=(1, 1, observation_size), dtype=np.float32)
    behavioral_policy(mock_states)
    policy(mock_states)

    trfl.update_target_variables(
        source_variables=policy.variables,
        target_variables=behavioral_policy.variables)

    states, actions, rewards, next_states, weights = rollout(
        exploration_wrapper, episodes=params.episodes, render=False)
    states_normalizer(states, training=True)
    actions_normalizer(actions, training=True)
    rewards_normalizer(rewards, training=True)

    for it in trange(params.iters):
        states, actions, rewards, next_states, weights = rollout(
            exploration_wrapper, episodes=params.episodes, render=False)

        episodic_rewards = tf.reduce_mean(tf.reduce_sum(rewards, axis=-1))

        with tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar('episodic_rewards/train',
                                      episodic_rewards)
            tf.contrib.summary.histogram('actions/train', actions)
            tf.contrib.summary.histogram('rewards/train', rewards)

        dataset = create_dataset(
            tensors=(states, actions, rewards, next_states, weights),
            batch_size=params.batch_size)

        for epoch in range(params.epochs):
            for (states, actions, rewards, next_states, weights) in dataset:
                states_norm = states_normalizer(states, training=True)
                actions_norm = actions_normalizer(actions, training=True)
                rewards_norm = rewards_normalizer(rewards, training=True)

                values = value(states_norm, training=False)

                advantages = targets.compute_advantages(
                    rewards_norm,
                    values,
                    discount_factor=params.discount_factor,
                    lambda_factor=params.lambda_factor,
                    weights=weights,
                    normalize=True)
                returns = targets.compute_returns(
                    rewards_norm,
                    discount_factor=params.discount_factor,
                    weights=weights)

                behavioral_policy_dist = behavioral_policy(
                    states_norm, training=False)

                log_probs_old = behavioral_policy_dist.log_prob(actions_norm)
                log_probs_old = tf.check_numerics(log_probs_old,
                                                  'log_probs_old')

                with tf.GradientTape() as tape:
                    policy_dist = policy(states_norm, training=True)

                    log_probs = policy_dist.log_prob(actions_norm)
                    log_probs = tf.check_numerics(log_probs, 'log_probs')

                    entropy = policy_dist.entropy()
                    entropy = tf.check_numerics(entropy, 'entropy')

                    values = value(states_norm, training=True)

                    # losses
                    policy_loss = losses.policy_ratio_loss(
                        log_probs=log_probs,
                        log_probs_old=tf.stop_gradient(log_probs_old),
                        advantages=advantages,
                        weights=weights,
                        epsilon_clipping=params.epsilon_clipping)
                    value_loss = params.value_coef * (
                        tf.losses.mean_squared_error(
                            predictions=values,
                            labels=returns,
                            weights=weights))
                    entropy_loss = -params.entropy_coef * (
                        tf.losses.compute_weighted_loss(
                            losses=entropy, weights=weights))
                    loss = policy_loss + value_loss + entropy_loss

                trainable_variables = (
                    policy.trainable_variables + value.trainable_variables)
                grads = tape.gradient(loss, trainable_variables)
                grads_clipped, grads_norm = tf.clip_by_global_norm(
                    grads, params.grad_clipping)
                grads_clipped_norm = tf.global_norm(grads_clipped)
                grads_and_vars = zip(grads_clipped, trainable_variables)
                optimizer.apply_gradients(
                    grads_and_vars, global_step=global_step)

                kl = tfp.distributions.kl_divergence(policy_dist,
                                                     behavioral_policy_dist)

                with tf.contrib.summary.always_record_summaries():
                    tf.contrib.summary.scalar('scale_diag', policy.scale_diag)
                    tf.contrib.summary.scalar('entropy', entropy)
                    tf.contrib.summary.scalar('kl', kl)
                    tf.contrib.summary.scalar('losses/policy_loss',
                                              policy_loss)
                    tf.contrib.summary.scalar('losses/entropy', entropy_loss)
                    tf.contrib.summary.scalar('losses/value_loss', value_loss)
                    tf.contrib.summary.scalar('grads_norm', grads_norm)
                    tf.contrib.summary.scalar('grads_norm/clipped',
                                              grads_clipped_norm)

        trfl.update_target_variables(
            source_variables=policy.trainable_variables,
            target_variables=behavioral_policy.trainable_variables)

        states, actions, rewards, next_states, weights = rollout(
            inference_wrapper, episodes=params.episodes, render=args.render)
        episodic_rewards = tf.reduce_mean(tf.reduce_sum(rewards, axis=-1))
        with tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar('episodic_rewards/eval',
                                      episodic_rewards)
            tf.contrib.summary.histogram('actions/eval', actions)
            tf.contrib.summary.histogram('rewards/eval', rewards)

        checkpoint_prefix = os.path.join(args.job_dir, 'ckpt')
        checkpoint.save(file_prefix=checkpoint_prefix)


if __name__ == '__main__':
    main()
