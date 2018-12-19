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
from tool_use.batch_env import BatchEnv
from tool_use.normalizer import HighLowNormalizer, Normalizer
from tool_use.parallel_rollout import ParallelRollout


class PolicyWrapper:
    def __init__(self, policy, states_normalizer):
        self.policy = policy
        self.states_normalizer = states_normalizer

    def __call__(self, state, *args, **kwargs):
        state = tf.convert_to_tensor(state, dtype=np.float32)
        state_batch = tf.expand_dims(state, axis=1)
        state_batch_norm = self.states_normalizer(state_batch)
        action_batch = self.policy(state_batch_norm, *args, **kwargs)
        action = tf.squeeze(action_batch, axis=1)
        action = action.numpy()
        return action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', required=True)
    parser.add_argument('--seed', default=42)
    args = parser.parse_args()
    print(args)

    # params
    params = HyperParams()
    print(params)

    # eager
    tf.enable_eager_execution()

    # environment
    env = BatchEnv(PendulumEnv, batch_size=params.episodes, blocking=False)
    state_size = env.observation_space.shape[0]

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
    policy_anchor = Policy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        scale=params.scale)
    value = Value(observation_space=env.observation_space)

    # normalization
    states_normalizer = HighLowNormalizer(
        low=env.observation_space.low,
        high=env.observation_space.high,
        alpha=1,
        center=True,
        scale=True)
    rewards_normalizer = Normalizer(shape=[], center=False, scale=True)

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
    rollout = ParallelRollout(env, max_episode_steps=params.max_episode_steps)

    # strategies
    exploration_strategy = PolicyWrapper(
        pyrl.strategies.SampleStrategy(policy),
        states_normalizer=states_normalizer)
    inference_strategy = PolicyWrapper(
        pyrl.strategies.ModeStrategy(policy),
        states_normalizer=states_normalizer)

    # prime models
    mock_states = tf.zeros(shape=(1, 1, state_size), dtype=np.float32)
    policy(mock_states, reset_state=True)
    policy_anchor(mock_states, reset_state=True)

    # sync variables
    trfl.update_target_variables(
        source_variables=policy.variables,
        target_variables=policy_anchor.variables)

    for it in trange(params.iters):
        # training
        transitions = rollout(exploration_strategy)

        dataset = tf.data.Dataset.from_tensors(transitions)

        for (states, actions, rewards, next_states, weights) in dataset:
            states_norm = states_normalizer(states, training=False)
            rewards_norm = rewards_normalizer(rewards, training=True)

            values = value(states_norm)

            advantages = targets.compute_advantages(
                rewards=rewards_norm,
                values=values,
                discount_factor=params.discount_factor,
                lambda_factor=params.lambda_factor,
                weights=weights,
                normalize=True)
            # returns = targets.compute_returns(
            #     rewards=rewards_norm,
            #     discount_factor=params.discount_factor,
            #     weights=weights)
            returns = tf.stop_gradient(advantages + values)

            policy_anchor_dist = policy_anchor(states_norm, reset_state=True)

            log_probs_anchor = policy_anchor_dist.log_prob(actions)
            log_probs_anchor = tf.check_numerics(log_probs_anchor,
                                                 'log_probs_anchor')

            episodic_rewards = tf.reduce_mean(tf.reduce_sum(rewards, axis=-1))

            with tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar('episodic_rewards/train',
                                          episodic_rewards)
                tf.contrib.summary.histogram('states/train', states)
                tf.contrib.summary.histogram('states_norm/train', states_norm)
                tf.contrib.summary.histogram('actions/train', actions)
                tf.contrib.summary.histogram('rewards/train', rewards)
                tf.contrib.summary.histogram('returns/train', returns)
                tf.contrib.summary.histogram('advantages/train', advantages)
                tf.contrib.summary.histogram('rewards_norm/train',
                                             rewards_norm)

                tf.contrib.summary.scalar(
                    'states_normalizer/mean',
                    tf.reduce_mean(states_normalizer.mean))
                tf.contrib.summary.scalar(
                    'states_normalizer/std',
                    tf.reduce_mean(states_normalizer.std))

                tf.contrib.summary.scalar(
                    'rewards_normalizer/mean',
                    tf.reduce_mean(rewards_normalizer.mean))
                tf.contrib.summary.scalar(
                    'rewards_normalizer/std',
                    tf.reduce_mean(rewards_normalizer.std))

            for epoch in range(params.epochs):
                with tf.GradientTape() as tape:
                    policy_dist = policy(
                        states_norm, training=True, reset_state=True)

                    log_probs = policy_dist.log_prob(actions)
                    log_probs = tf.check_numerics(log_probs, 'log_probs')

                    entropy = policy_dist.entropy()
                    entropy = tf.check_numerics(entropy, 'entropy')

                    values = value(states_norm, training=True)

                    # losses
                    policy_loss = losses.policy_ratio_loss(
                        log_probs=log_probs,
                        log_probs_anchor=log_probs_anchor,
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
        if it % params.eval_interval == 0:
            states, actions, rewards, next_states, weights = rollout(
                inference_strategy)
            episodic_rewards = tf.reduce_mean(tf.reduce_sum(rewards, axis=-1))

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
