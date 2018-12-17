import os
import trfl
import random
import argparse
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import tensorflow_probability as tfp

import pyoneer.rl as pyrl

from tqdm import trange
# from gym.envs.classic_control import PendulumEnv

from tool_use import losses
from tool_use import targets
from tool_use.env import KukaEnv
from tool_use.models import Policy, Value, StateModel
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
    # env = PendulumEnv()
    # env = KukaEnv()
    env = BatchEnv(KukaEnv, batch_size=params.episodes, blocking=False)

    # seeding
    env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # optimization
    global_step = tf.train.create_global_step()
    state_global_step = tfe.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
    state_optimizer = tf.train.AdamOptimizer(
        learning_rate=params.learning_rate)

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
    state_model = StateModel(observation_space=env.observation_space)
    state_model_rand = StateModel(observation_space=env.observation_space)

    # normalization
    extrinsic_rewards_normalizer = Normalizer(
        shape=(), center=False, scale=True)
    intrinsic_rewards_normalizer = Normalizer(
        shape=(), center=False, scale=True)

    # checkpoints
    checkpoint = tf.train.Checkpoint(
        global_step=global_step,
        state_global_step=state_global_step,
        optimizer=optimizer,
        state_optimizer=state_optimizer,
        policy=policy,
        value=value,
        extrinsic_rewards_normalizer=extrinsic_rewards_normalizer)
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
    exploration_strategy = pyrl.strategies.SampleStrategy(policy)
    inference_strategy = pyrl.strategies.ModeStrategy(policy)

    exploration_strategy = PolicyWrapper(exploration_strategy)
    inference_strategy = PolicyWrapper(inference_strategy)

    # prime models
    mock_states = tf.zeros(
        shape=(1, 1, env.observation_space.shape[0]), dtype=np.float32)
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

        for (states, actions, extrinsic_rewards, next_states,
             weights) in dataset:
            state_model_rand_pred = state_model_rand(states)

            with tf.GradientTape() as tape:
                state_model_pred = state_model(states)
                loss = tf.losses.mean_squared_error(
                    predictions=state_model_pred,
                    labels=tf.stop_gradient(state_model_rand_pred),
                    weights=weights[..., None])

            with tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar('losses/state_model', loss)

            grads = tape.gradient(loss, state_model.trainable_variables)
            grads_clipped, grads_norm = tf.clip_by_global_norm(
                grads, params.grad_clipping)
            grads_clipped_norm = tf.global_norm(grads_clipped)
            grads_and_vars = zip(grads_clipped,
                                 state_model.trainable_variables)
            state_optimizer.apply_gradients(
                grads_and_vars, global_step=state_global_step)

            intrinsic_rewards = tf.reduce_sum(
                tf.squared_difference(state_model_pred, state_model_rand_pred),
                axis=-1)
            extrinsic_rewards_norm = extrinsic_rewards_normalizer(
                extrinsic_rewards, training=True)
            intrinsic_rewards_norm = intrinsic_rewards_normalizer(
                intrinsic_rewards, training=True)
            rewards_norm = extrinsic_rewards_norm + intrinsic_rewards_norm

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

            policy_anchor_dist = policy_anchor(states, reset_state=True)

            log_probs_anchor = policy_anchor_dist.log_prob(actions)
            log_probs_anchor = tf.check_numerics(log_probs_anchor,
                                                 'log_probs_anchor')

            episodic_extrinsic_rewards = tf.reduce_mean(
                tf.reduce_sum(extrinsic_rewards, axis=-1))
            episodic_intrinsic_rewards = tf.reduce_mean(
                tf.reduce_sum(intrinsic_rewards, axis=-1))

            with tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar('episodic_extrinsic_rewards/train',
                                          episodic_extrinsic_rewards)
                tf.contrib.summary.scalar('episodic_intrinsic_rewards/train',
                                          episodic_intrinsic_rewards)
                tf.contrib.summary.histogram('actions/train', actions)
                tf.contrib.summary.histogram('extrinsic_rewards/train',
                                             extrinsic_rewards)
                tf.contrib.summary.histogram('intrinsic_rewards/train',
                                             intrinsic_rewards)

                tf.contrib.summary.scalar(
                    'extrinsic_rewards_normalizer/mean',
                    tf.reduce_mean(extrinsic_rewards_normalizer.mean))
                tf.contrib.summary.scalar(
                    'extrinsic_rewards_normalizer/std',
                    tf.reduce_mean(extrinsic_rewards_normalizer.std))
                tf.contrib.summary.scalar(
                    'intrinsic_rewards_normalizer/mean',
                    tf.reduce_mean(intrinsic_rewards_normalizer.mean))
                tf.contrib.summary.scalar(
                    'intrinsic_rewards_normalizer/std',
                    tf.reduce_mean(intrinsic_rewards_normalizer.std))

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
        states, actions, rewards, next_states, weights = rollout(
            inference_strategy)
        episodic_rewards = tf.reduce_mean(tf.reduce_sum(rewards, axis=-1))

        with tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar('episodic_extrinsic_rewards/eval',
                                      episodic_rewards)
            tf.contrib.summary.histogram('actions/eval', actions)
            tf.contrib.summary.histogram('extrinsic_rewards/eval', rewards)

        # save checkpoint
        checkpoint_prefix = os.path.join(args.job_dir, 'ckpt')
        checkpoint.save(file_prefix=checkpoint_prefix)


if __name__ == '__main__':
    main()
