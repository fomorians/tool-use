import os
import gym
import random
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import pyoneer.rl as pyrl

from tqdm import trange
# from gym.envs.classic_control import PendulumEnv

from tool_use import losses
from tool_use import targets
from tool_use.models import Policy, Value
from tool_use.params import HyperParams
from tool_use.rollout import Rollout
from tool_use.normalizer import Normalizer


def copy_variables(source_vars, dest_vars):
    assert len(source_vars) == len(dest_vars), 'vars must be the same length'
    for dest_var, source_var in zip(dest_vars, source_vars):
        dest_var.assign(source_var)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', required=True)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--seed', default=67)
    args = parser.parse_args()
    print(args)

    # eager
    tf.enable_eager_execution()

    # environment
    env = gym.make('Pendulum-v0')
    # env = PendulumEnv()
    # env = KukaEnv(render=args.render)

    # params
    params = HyperParams()
    print(params)

    # seeding
    env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # optimization
    global_step = tf.train.create_global_step()
    optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)

    # models
    policy = Policy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        scale=params.scale)
    policy_old = Policy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        scale=params.scale)
    value = Value(observation_space=env.observation_space)

    # rewards
    rewards_normalizer = Normalizer(shape=(), center=False, scale=False)

    # checkpoints
    checkpoint = tf.train.Checkpoint(
        global_step=global_step,
        optimizer=optimizer,
        policy=policy,
        value=value,
        rewards_normalizer=rewards_normalizer)
    checkpoint_path = tf.train.latest_checkpoint(args.job_dir)
    if checkpoint_path is not None:
        checkpoint.restore(checkpoint_path)

    # summaries
    summary_writer = tf.contrib.summary.create_file_writer(
        args.job_dir, max_queue=1, flush_millis=1000)
    summary_writer.set_as_default()

    # rollouts
    rollout = Rollout(env, max_episode_steps=params.max_episode_steps)

    # strategies
    exploration_strategy = pyrl.strategies.SampleStrategy(policy)
    inference_strategy = pyrl.strategies.ModeStrategy(policy)

    # prime models
    mock_states = tf.zeros(
        shape=(1, 1, env.observation_space.shape[0]), dtype=np.float32)
    policy(mock_states)
    policy_old(mock_states)

    for it in trange(params.iters):
        # training
        states, actions, rewards, next_states, weights = rollout(
            exploration_strategy, episodes=params.episodes, render=False)

        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        rewards = tf.convert_to_tensor(rewards)
        next_states = tf.convert_to_tensor(next_states)
        weights = tf.convert_to_tensor(weights)

        rewards_norm = rewards_normalizer(rewards, training=True)

        returns = targets.compute_returns(
            rewards_norm,
            discount_factor=params.discount_factor,
            weights=weights)

        values = value(states)

        advantages = targets.compute_advantages(
            rewards=rewards_norm,
            values=values,
            discount_factor=params.discount_factor,
            lambda_factor=params.lambda_factor,
            weights=weights,
            normalize=True)

        copy_variables(
            source_vars=policy.trainable_variables,
            dest_vars=policy_old.trainable_variables)

        policy_old_dist = policy_old(states)

        log_probs_old = policy_old_dist.log_prob(actions)
        log_probs_old = tf.check_numerics(log_probs_old, 'log_probs_old')

        episodic_rewards = tf.reduce_mean(tf.reduce_sum(rewards, axis=-1))

        with tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar('episodic_rewards/train',
                                      episodic_rewards)
            tf.contrib.summary.histogram('actions/train', actions)
            tf.contrib.summary.histogram('rewards/train', rewards)

            tf.contrib.summary.scalar('rewards_normalizer/mean',
                                      tf.reduce_mean(rewards_normalizer.mean))
            tf.contrib.summary.scalar('rewards_normalizer/std',
                                      tf.reduce_mean(rewards_normalizer.std))

        for epoch in range(params.epochs):
            with tf.GradientTape() as tape:
                policy_dist = policy(states, training=True)

                log_probs = policy_dist.log_prob(actions)
                log_probs = tf.check_numerics(log_probs, 'log_probs')

                entropy = policy_dist.entropy()
                entropy = tf.check_numerics(entropy, 'entropy')

                values = value(states, training=True)

                # losses
                policy_loss = losses.policy_ratio_loss(
                    log_probs=log_probs,
                    log_probs_old=log_probs_old,
                    advantages=advantages,
                    weights=weights,
                    epsilon_clipping=params.epsilon_clipping)
                value_loss = params.value_coef * (tf.losses.mean_squared_error(
                    predictions=values, labels=returns, weights=weights))
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
            optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            kl = tfp.distributions.kl_divergence(policy_dist, policy_old_dist)
            entropy_mean = tf.losses.compute_weighted_loss(
                losses=entropy, weights=weights)

            with tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar('scale_diag', policy.scale_diag)
                tf.contrib.summary.scalar('entropy', entropy_mean)
                tf.contrib.summary.scalar('kl', kl)
                tf.contrib.summary.scalar('losses/policy_loss', policy_loss)
                tf.contrib.summary.scalar('losses/entropy', entropy_loss)
                tf.contrib.summary.scalar('losses/value_loss', value_loss)
                tf.contrib.summary.scalar('grads_norm', grads_norm)
                tf.contrib.summary.scalar('grads_norm/clipped',
                                          grads_clipped_norm)

        # evaluation
        states, actions, rewards, next_states, weights = rollout(
            inference_strategy, episodes=params.episodes, render=args.render)
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
