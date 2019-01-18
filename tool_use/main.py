import os
import gym
import random
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import pyoneer as pynr
import pyoneer.rl as pyrl

from tqdm import trange

from tool_use.models import Policy, Value
from tool_use.params import HyperParams
from tool_use.parallel_rollout import ParallelRollout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', required=True)
    parser.add_argument('--env', default='Pendulum-v0')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--train-iters', default=100, type=int)
    args = parser.parse_args()
    print(args)

    # register kuka env
    # gym.envs.register(
    #     id='KukaEnv-v0',
    #     entry_point='tool_use.kuka_env:KukaEnv',
    #     max_episode_steps=1000)
    # gym.make('KukaEnv-v0')

    # make job directory
    if not os.path.exists(args.job_dir):
        os.makedirs(args.job_dir)

    # params
    params = HyperParams(
        env=args.env, seed=args.seed, train_iters=args.train_iters)
    params_path = os.path.join(args.job_dir, 'params.json')
    params.save(params_path)
    print(params)

    # eager
    tf.enable_eager_execution()

    def env_constructor():
        return gym.make(params.env)

    # environment
    env = pyrl.envs.BatchEnv(
        constructor=env_constructor,
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
    baseline = Value(observation_space=env.observation_space)

    # strategies
    exploration_strategy = pyrl.strategies.SampleStrategy(policy)
    inference_strategy = pyrl.strategies.ModeStrategy(policy)

    # normalization
    rewards_moments = pynr.nn.ExponentialMovingMoments(
        shape=(), rate=params.reward_decay)

    # checkpoints
    checkpoint = tf.train.Checkpoint(
        global_step=global_step,
        optimizer=optimizer,
        policy=policy,
        baseline=baseline,
        rewards_moments=rewards_moments)
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

    # prime models
    # NOTE: TF eager does not initialize weights until they're called
    mock_states = tf.zeros(
        shape=(1, 1, env.observation_space.shape[0]), dtype=np.float32)
    policy(mock_states, reset_state=True)
    policy_anchor(mock_states, reset_state=True)

    for it in trange(params.train_iters):
        # training
        transitions = rollout(exploration_strategy)
        dataset = tf.data.Dataset.from_tensors(transitions)

        for (states, actions, rewards, next_states, weights) in dataset:
            rewards_moments(rewards, weights=weights, training=True)
            rewards_norm = pynr.math.normalize(
                rewards,
                loc=rewards_moments.mean,
                scale=rewards_moments.std,
                weights=weights)

            values_target = baseline(states, reset_state=True)

            advantages = pyrl.targets.generalized_advantages(
                rewards=rewards_norm,
                values=values_target,
                discount_factor=params.discount_factor,
                lambda_factor=params.lambda_factor,
                weights=weights,
                normalize=True)
            returns = pyrl.targets.discounted_rewards(
                rewards=rewards_norm,
                discount_factor=params.discount_factor,
                weights=weights)

            # sync variables
            pynr.training.update_target_variables(
                source_variables=policy.variables,
                target_variables=policy_anchor.variables)

            policy_anchor_dist = policy_anchor(states, reset_state=True)
            log_probs_anchor = policy_anchor_dist.log_prob(actions)

            with tf.contrib.summary.always_record_summaries():
                episodic_rewards = tf.reduce_mean(
                    tf.reduce_sum(rewards, axis=-1))

                tf.contrib.summary.scalar('episodic_rewards/train',
                                          episodic_rewards)
                tf.contrib.summary.scalar('rewards/mean', rewards_moments.mean)
                tf.contrib.summary.scalar('rewards/std', rewards_moments.std)

                for i in range(env.observation_space.shape[0]):
                    tf.contrib.summary.histogram(f'states/{i}/train',
                                                 states[..., i])
                for i in range(env.action_space.shape[0]):
                    tf.contrib.summary.histogram(f'actions/{i}/train',
                                                 actions[..., i])
                tf.contrib.summary.histogram('rewards/train', rewards)

            # training epochs
            for epoch in range(params.epochs):
                with tf.GradientTape() as tape:
                    # forward passes
                    policy_dist = policy(
                        states, training=True, reset_state=True)
                    log_probs = policy_dist.log_prob(actions)
                    entropy = policy_dist.entropy()
                    values = baseline(states, training=True, reset_state=True)

                    # losses
                    policy_loss = pyrl.losses.clipped_policy_gradient_loss(
                        log_probs=log_probs,
                        log_probs_anchor=log_probs_anchor,
                        advantages=advantages,
                        epsilon_clipping=params.epsilon_clipping,
                        weights=weights)
                    value_loss = tf.losses.mean_squared_error(
                        predictions=values,
                        labels=returns,
                        weights=weights * params.value_coef)
                    entropy_loss = -tf.losses.compute_weighted_loss(
                        losses=entropy, weights=weights * params.entropy_coef)
                    loss = policy_loss + value_loss + entropy_loss

                # optimization
                trainable_variables = (
                    policy.trainable_variables + baseline.trainable_variables)
                grads = tape.gradient(loss, trainable_variables)
                if params.grad_clipping is not None:
                    grads_clipped, _ = tf.clip_by_global_norm(
                        grads, params.grad_clipping)
                grads_and_vars = zip(grads_clipped, trainable_variables)
                optimizer.apply_gradients(
                    grads_and_vars, global_step=global_step)

                with tf.contrib.summary.always_record_summaries():
                    kl = tfp.distributions.kl_divergence(
                        policy_dist, policy_anchor_dist)
                    entropy_mean = tf.losses.compute_weighted_loss(
                        losses=entropy, weights=weights)

                    tf.contrib.summary.scalar('losses/policy', policy_loss)
                    tf.contrib.summary.scalar('losses/value', value_loss)
                    tf.contrib.summary.scalar('losses/entropy', entropy_loss)
                    tf.contrib.summary.scalar('losses/loss', loss)

                    tf.contrib.summary.scalar('gradient_norm/unclipped',
                                              tf.global_norm(grads))
                    tf.contrib.summary.scalar('gradient_norm',
                                              tf.global_norm(grads_clipped))

                    tf.contrib.summary.scalar('scale_diag', policy.scale_diag)
                    tf.contrib.summary.scalar('entropy', entropy_mean)
                    tf.contrib.summary.scalar('kl', kl)

        # evaluation
        if it % params.eval_interval == params.eval_interval - 1:
            states, actions, rewards, next_states, weights = rollout(
                inference_strategy)

            with tf.contrib.summary.always_record_summaries():
                episodic_rewards = tf.reduce_mean(
                    tf.reduce_sum(rewards, axis=-1))

                tf.contrib.summary.scalar('episodic_rewards/eval',
                                          episodic_rewards)

                for i in range(env.observation_space.shape[0]):
                    tf.contrib.summary.histogram(f'states/{i}/eval',
                                                 states[..., i])
                for i in range(env.action_space.shape[0]):
                    tf.contrib.summary.histogram(f'actions/{i}/eval',
                                                 actions[..., i])
                tf.contrib.summary.histogram('rewards/eval', rewards)

            # save checkpoint
            checkpoint_prefix = os.path.join(args.job_dir, 'ckpt')
            checkpoint.save(file_prefix=checkpoint_prefix)


if __name__ == '__main__':
    main()
