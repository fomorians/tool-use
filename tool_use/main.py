import os
import json
import trfl
import random
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import pyoneer as pynr
import pyoneer.rl as pyrl

from tqdm import trange

from tool_use import losses
from tool_use import targets
from tool_use.models import Policy, Value
from tool_use.params import HyperParams
from tool_use.batch_env import BatchEnv
from tool_use.normalizer import Normalizer
from tool_use.parallel_rollout import ParallelRollout


class PolicyWrapper:
    def __init__(self, policy):
        self.policy = policy

    def __call__(self, state, *args, **kwargs):
        state = tf.convert_to_tensor(state, dtype=np.float32)
        state_batch = tf.expand_dims(state, axis=1)
        action_batch = self.policy(state_batch, *args, **kwargs)
        action = tf.squeeze(action_batch, axis=1)
        action = action.numpy()
        return action


def run_experiment(job_dir, env_name, seed, use_discount):
    # make job directory
    if not os.path.exists(job_dir):
        os.makedirs(job_dir)

    # params
    params = HyperParams()
    params_path = os.path.join(job_dir, 'params.json')
    params.save(params_path)
    print(params)

    # eager
    tf.enable_eager_execution()

    # environment
    env = BatchEnv(env_name, batch_size=params.episodes, blocking=False)
    state_size = env.observation_space.shape[0]

    # seeding
    env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

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
    value = Value(observation_space=env.observation_space)

    # normalization
    rewards_normalizer = Normalizer(shape=[], center=False, scale=True)

    # checkpoints
    checkpoint = tf.train.Checkpoint(
        global_step=global_step,
        optimizer=optimizer,
        policy=policy,
        value=value,
        rewards_normalizer=rewards_normalizer)
    checkpoint_path = tf.train.latest_checkpoint(job_dir)
    if checkpoint_path is not None:
        checkpoint.restore(checkpoint_path)

    # summaries
    summary_writer = tf.contrib.summary.create_file_writer(
        job_dir, max_queue=1, flush_millis=1000)
    summary_writer.set_as_default()

    # rollouts
    rollout = ParallelRollout(
        env, max_episode_steps=env.spec.max_episode_steps)

    # strategies
    exploration_strategy = PolicyWrapper(
        pyrl.strategies.SampleStrategy(policy))
    inference_strategy = PolicyWrapper(pyrl.strategies.ModeStrategy(policy))

    # prime models
    mock_states = tf.zeros(shape=(1, 1, state_size), dtype=np.float32)
    policy(mock_states, reset_state=True)
    policy_anchor(mock_states, reset_state=True)

    # sync variables
    trfl.update_target_variables(
        source_variables=policy.variables,
        target_variables=policy_anchor.variables)

    reward_history = []

    # evaluation
    states, actions, rewards, next_states, weights = rollout(
        inference_strategy)
    episodic_rewards = tf.reduce_mean(
        tf.reduce_sum(rewards * weights, axis=-1))
    reward_history.append(float(episodic_rewards.numpy()))

    with tf.contrib.summary.always_record_summaries():
        tf.contrib.summary.scalar('episodic_rewards/eval', episodic_rewards)
        tf.contrib.summary.histogram('actions/eval', actions)
        tf.contrib.summary.histogram('rewards/eval', rewards)

    for it in trange(params.iters):
        # training
        transitions = rollout(exploration_strategy)

        dataset = tf.data.Dataset.from_tensors(transitions)

        for (states, actions, rewards, next_states, weights) in dataset:
            rewards_norm = rewards_normalizer(rewards, weights, training=True)

            values_target = value(states)

            advantages = targets.compute_advantages(
                rewards=rewards_norm,
                values=values_target,
                discount_factor=params.discount_factor,
                lambda_factor=params.lambda_factor,
                weights=weights,
                normalize=False)
            advantages_norm = pynr.math.weighted_moments_normalize(
                advantages, weights=weights)
            if use_discount:
                returns = targets.compute_returns(
                    rewards=rewards_norm,
                    discount_factor=params.discount_factor,
                    weights=weights)
            else:
                returns = tf.stop_gradient(advantages_norm + values_target)

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
                tf.contrib.summary.histogram('advantages/train',
                                             advantages_norm)
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

                    values = value(states, training=True)

                    # losses
                    policy_loss = losses.policy_ratio_loss(
                        log_probs=log_probs,
                        log_probs_anchor=log_probs_anchor,
                        advantages=advantages_norm,
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
            reward_history.append(float(episodic_rewards.numpy()))

            with tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar('episodic_rewards/eval',
                                          episodic_rewards)
                tf.contrib.summary.histogram('actions/eval', actions)
                tf.contrib.summary.histogram('rewards/eval', rewards)

        # save checkpoint
        checkpoint_prefix = os.path.join(job_dir, 'ckpt')
        checkpoint.save(file_prefix=checkpoint_prefix)

    rewards_path = os.path.join(job_dir, 'rewards.json')
    with open(rewards_path, 'w') as fp:
        json.dump(reward_history, fp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', required=True)
    parser.add_argument('--env-name', default='Pendulum-v0')
    parser.add_argument('--use-discount', action='store_true')
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()
    print(args)

    run_experiment(
        job_dir=args.job_dir,
        env_name=args.env_name,
        seed=args.seed,
        use_discount=args.use_discount)


if __name__ == '__main__':
    main()
