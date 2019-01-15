import os
import trfl
import random
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import pyoneer.rl as pyrl

from tqdm import trange

from tool_use import losses
from tool_use import targets
from tool_use.models import Policy, Value
from tool_use.params import HyperParams
from tool_use.batch_env import BatchEnv
from tool_use.normalizer import MovingAverageNormalizer
from tool_use.parallel_rollout import ParallelRollout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', required=True)
    parser.add_argument('--env-name', default='Pendulum-v0')
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()
    print(args)

    # make job directory
    if not os.path.exists(args.job_dir):
        os.makedirs(args.job_dir)

    # params
    params = HyperParams(env_name=args.env_name, seed=args.seed)
    params_path = os.path.join(args.job_dir, 'params.json')
    params.save(params_path)
    print(params)

    # eager
    tf.enable_eager_execution()

    # environment
    env = pyrl.envs.BatchEnv(
        constructor=lambda: gym.make(params.env_name),
        batch_size=params.episodes,
        blocking=False)

    # seeding
    env.seed(params.seed)
    random.seed(params.seed)
    np.random.seed(params.seed)
    tf.set_random_seed(params.seed)

    # optimization
    global_step = tf.train.create_global_step()
    optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)

    # models
    policy = Policy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        scale=params.scale)
    policy_anchor = Policy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        scale=params.scale)
    value = Value(
        observation_space=env.observation_space, action_space=env.action_space)

    # normalization
    rewards_normalizer = MovingAverageNormalizer(
        shape=[], rate=params.reward_decay, center=False, scale=True)

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
        args.job_dir, max_queue=100, flush_millis=5 * 60 * 1000)
    summary_writer.set_as_default()

    # rollouts
    rollout = ParallelRollout(
        env, max_episode_steps=env.spec.max_episode_steps)

    # strategies
    exploration_strategy = pyrl.strategies.SampleStrategy(policy)
    inference_strategy = pyrl.strategies.ModeStrategy(policy)

    # prime models
    state_size = env.observation_space.shape[0]
    mock_states = tf.zeros(shape=(1, 1, state_size), dtype=np.float32)
    policy(mock_states, reset_state=True)
    policy_anchor(mock_states, reset_state=True)

    # sync variables
    trfl.update_target_variables(
        source_variables=policy.variables,
        target_variables=policy_anchor.variables)

    # evaluation
    states, actions, rewards, next_states, weights = rollout(
        inference_strategy)
    episodic_rewards = tf.reduce_mean(
        tf.reduce_sum(rewards * weights, axis=-1))

    with tf.contrib.summary.always_record_summaries():
        tf.contrib.summary.scalar('episodic_rewards/eval', episodic_rewards)
        tf.contrib.summary.histogram('actions/eval', actions)
        tf.contrib.summary.histogram('rewards/eval', rewards)

    for it in trange(params.train_iters):
        # training
        transitions = rollout(exploration_strategy)

        dataset = tf.data.Dataset.from_tensors(transitions)

        for (states, actions, rewards, next_states, weights) in dataset:
            rewards_norm = rewards_normalizer(rewards, weights, training=True)

            values_target = value(states, actions, reset_state=True)

            advantages = targets.generalized_advantages(
                rewards=rewards_norm,
                values=values_target,
                discount_factor=params.discount_factor,
                lambda_factor=params.lambda_factor,
                weights=weights,
                normalize=True)
            returns = targets.discounted_rewards(
                rewards=rewards_norm,
                discount_factor=params.discount_factor,
                weights=weights)

            policy_anchor_dist = policy_anchor(states, reset_state=True)

            log_probs_anchor = policy_anchor_dist.log_prob(actions)
            log_probs_anchor = tf.check_numerics(log_probs_anchor,
                                                 'log_probs_anchor')

            episodic_rewards = tf.reduce_mean(
                tf.reduce_sum(rewards * weights, axis=-1))

            with tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar('episodic_rewards/train',
                                          episodic_rewards)
                tf.contrib.summary.histogram('states/train', states)
                tf.contrib.summary.histogram('actions/train', actions)
                tf.contrib.summary.histogram('rewards/train', rewards)
                tf.contrib.summary.histogram('returns/train', returns)
                tf.contrib.summary.histogram('advantages/train', advantages)
                tf.contrib.summary.histogram('values/train', values_target)
                tf.contrib.summary.histogram('rewards_norm/train',
                                             rewards_norm)

                tf.contrib.summary.scalar(
                    'rewards_normalizer/mean',
                    tf.reduce_mean(rewards_normalizer.mean))
                tf.contrib.summary.scalar(
                    'rewards_normalizer/std',
                    tf.reduce_mean(rewards_normalizer.std))

            for epoch in range(params.epochs):
                with tf.GradientTape() as tape:
                    policy_dist = policy(
                        states, training=True, reset_state=True)

                    log_probs = policy_dist.log_prob(actions)
                    log_probs = tf.check_numerics(log_probs, 'log_probs')

                    entropy = policy_dist.entropy()
                    entropy = tf.check_numerics(entropy, 'entropy')

                    values = value(
                        states, actions, training=True, reset_state=True)

                    # losses
                    policy_loss = losses.policy_ratio_loss(
                        log_probs=log_probs,
                        log_probs_anchor=log_probs_anchor,
                        advantages=advantages,
                        weights=weights,
                        epsilon_clipping=params.epsilon_clipping)
                    value_loss = tf.losses.mean_squared_error(
                        predictions=values,
                        labels=returns,
                        weights=weights * params.value_coef)
                    entropy_loss = -tf.losses.compute_weighted_loss(
                        losses=entropy, weights=weights * params.entropy_coef)
                    loss = (policy_loss + value_loss + entropy_loss)

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
                                                     policy_anchor_dist)
                entropy_mean = tf.losses.compute_weighted_loss(
                    losses=entropy, weights=weights)

                with tf.contrib.summary.always_record_summaries():
                    tf.contrib.summary.scalar('scale_diag', policy.scale_diag)
                    tf.contrib.summary.scalar('entropy', entropy_mean)
                    tf.contrib.summary.scalar('kl', kl)
                    tf.contrib.summary.scalar('losses/policy_loss',
                                              policy_loss)
                    tf.contrib.summary.scalar('losses/entropy', entropy_loss)
                    tf.contrib.summary.scalar('losses/value_loss', value_loss)
                    tf.contrib.summary.scalar('grads_norm', grads_norm)
                    tf.contrib.summary.scalar('grads_norm/clipped',
                                              grads_clipped_norm)

        # update anchor
        trfl.update_target_variables(
            source_variables=policy.trainable_variables,
            target_variables=policy_anchor.trainable_variables)

        # evaluation
        if it % params.eval_interval == params.eval_interval - 1:
            states, actions, rewards, next_states, weights = rollout(
                inference_strategy)
            episodic_rewards = tf.reduce_mean(
                tf.reduce_sum(rewards * weights, axis=-1))

            with tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar('episodic_rewards/eval',
                                          episodic_rewards)
                tf.contrib.summary.histogram('actions/eval', actions)
                tf.contrib.summary.histogram('rewards/eval', rewards)

            # save checkpoint
            checkpoint_prefix = os.path.join(args.job_dir, 'ckpt')
            checkpoint.save(file_prefix=checkpoint_prefix)


if __name__ == '__main__':
    main()
