import os
import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import pyoneer as pynr
import pyoneer.rl as pyrl

from tool_use import models
from tool_use.wrappers import RangeNormalize
from tool_use.parallel_rollout import ParallelRollout


class Trainer:
    def __init__(self, job_dir, params, include_histograms=False):
        self.params = params

        # environment
        def env_constructor():
            env = gym.make(self.params.env)
            env = RangeNormalize(env)
            return env

        self.env = pyrl.envs.BatchEnv(
            constructor=env_constructor,
            batch_size=self.params.num_envs,
            blocking=False)

        observation_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.shape[0]

        # seeding
        self.env.seed(self.params.seed)

        # optimization
        self.global_step = tf.train.create_global_step()
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.params.learning_rate)

        # models
        self.policy = models.PolicyValue(
            action_size=action_size, scale=self.params.scale)
        self.policy_anchor = models.PolicyValue(
            action_size=action_size, scale=self.params.scale)

        # strategies
        self.exploration_strategy = pyrl.strategies.SampleStrategy(self.policy)
        self.inference_strategy = pyrl.strategies.ModeStrategy(self.policy)

        # normalization
        self.rewards_moments = pynr.nn.ExponentialMovingMoments(
            shape=(), rate=self.params.reward_decay)

        # checkpoints
        self.checkpoint = tf.train.Checkpoint(
            global_step=self.global_step,
            optimizer=self.optimizer,
            policy=self.policy,
            rewards_moments=self.rewards_moments)
        checkpoint_path = tf.train.latest_checkpoint(self.job_dir)
        if checkpoint_path is not None:
            self.checkpoint.restore(checkpoint_path)

        # summaries
        summary_writer = tf.contrib.summary.create_file_writer(
            self.job_dir, max_queue=100, flush_millis=5 * 60 * 1000)
        summary_writer.set_as_default()

        # rollouts
        self.rollout = ParallelRollout(
            self.env, max_episode_steps=self.env.spec.max_episode_steps)

        # prime models
        # NOTE: TF eager does not initialize weights until they're called
        mock_observations = tf.zeros(
            shape=(1, 1, observation_size), dtype=np.float32)
        self.policy(mock_observations, reset_state=True)
        self.policy_anchor(mock_observations, reset_state=True)
        self.policy.get_values(mock_observations, reset_state=True)
        self.policy_anchor.get_values(mock_observations, reset_state=True)

    def _train(self):
        # sync variables
        pynr.training.update_target_variables(
            source_variables=self.policy.variables,
            target_variables=self.policy_anchor.variables)

        # training
        states, actions, rewards, next_states, weights = self.rollout(
            self.exploration_strategy, episodes=self.params.episodes)

        miniepisodes = (self.params.episodes *
                        self.env.spec.max_episode_steps) // self.params.horizon

        states = states.reshape(miniepisodes, self.params.horizon,
                                states.shape[-1])
        actions = actions.reshape(miniepisodes, self.params.horizon,
                                  actions.shape[-1])
        rewards = rewards.reshape(miniepisodes, self.params.horizon)
        next_states = next_states.reshape(miniepisodes, self.params.horizon,
                                          next_states.shape[-1])
        weights = weights.reshape(miniepisodes, self.params.horizon)

        self.rewards_moments(rewards, weights=weights, training=True)

        rewards_norm = pynr.math.safe_divide(rewards, self.rewards_moments.std)

        values_target = self.policy.get_values(states, reset_state=True)

        advantages = pyrl.targets.generalized_advantages(
            rewards=rewards_norm,
            values=values_target,
            discount_factor=self.params.discount_factor,
            lambda_factor=self.params.lambda_factor,
            weights=weights,
            normalize=True)
        returns = pyrl.targets.discounted_rewards(
            rewards=rewards_norm,
            discount_factor=self.params.discount_factor,
            weights=weights)

        episodic_rewards = tf.reduce_mean(tf.reduce_sum(rewards, axis=-1))
        print(self.global_step.numpy(), 'episodic_rewards/train',
              episodic_rewards.numpy())

        with tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar('episodic_rewards/train',
                                      episodic_rewards)
            tf.contrib.summary.scalar('rewards/mean',
                                      self.rewards_moments.mean)
            tf.contrib.summary.scalar('rewards/std', self.rewards_moments.std)

            if self.include_histograms:
                for i in range(self.env.observation_space.shape[0]):
                    tf.contrib.summary.histogram('states/{}/train'.format(i),
                                                 states[..., i])
                for i in range(self.env.action_space.shape[0]):
                    tf.contrib.summary.histogram('actions/{}/train'.format(i),
                                                 actions[..., i])
                tf.contrib.summary.histogram('rewards/train', rewards)
                tf.contrib.summary.histogram('rewards_norm/train', rewards)
                tf.contrib.summary.histogram('rewards_norm/train',
                                             rewards_norm)
                tf.contrib.summary.histogram('advantages/train', advantages)
                tf.contrib.summary.histogram('returns/train', returns)

        with tf.device('/cpu:0'):
            dataset = tf.data.Dataset.from_tensor_slices(
                (states, actions, rewards_norm, advantages, returns, weights))
            dataset = dataset.batch(self.params.batch_size)
            dataset = dataset.repeat(self.params.epochs)
            dataset = dataset.apply(
                tf.data.experimental.prefetch_to_device('/gpu:0'))

        for (states, actions, rewards_norm, advantages, returns,
             weights) in dataset:
            with tf.GradientTape() as tape:
                # forward passes
                policy_dist = self.policy(
                    states, training=True, reset_state=True)
                log_probs = policy_dist.log_prob(actions)

                policy_anchor_dist = self.policy_anchor(
                    states, reset_state=True)
                log_probs_anchor = policy_anchor_dist.log_prob(actions)

                entropy = policy_dist.entropy()
                values = self.policy.get_values(
                    states, training=True, reset_state=True)

                # losses
                policy_loss = pyrl.losses.clipped_policy_gradient_loss(
                    log_probs=log_probs,
                    log_probs_anchor=log_probs_anchor,
                    advantages=advantages,
                    epsilon_clipping=self.params.epsilon_clipping,
                    weights=weights)
                value_loss = tf.losses.mean_squared_error(
                    predictions=values,
                    labels=returns,
                    weights=weights * self.params.value_coef)
                entropy_loss = -tf.losses.compute_weighted_loss(
                    losses=entropy, weights=weights * self.params.entropy_coef)
                loss = policy_loss + value_loss + entropy_loss

            # optimization
            grads = tape.gradient(loss, self.policy.trainable_variables)
            if self.params.grad_clipping is not None:
                grads_clipped, _ = tf.clip_by_global_norm(
                    grads, self.params.grad_clipping)
            grads_and_vars = zip(grads_clipped,
                                 self.policy.trainable_variables)
            self.optimizer.apply_gradients(
                grads_and_vars, global_step=self.global_step)

            with tf.contrib.summary.always_record_summaries():
                kl = tfp.distributions.kl_divergence(policy_dist,
                                                     policy_anchor_dist)
                entropy_mean = tf.losses.compute_weighted_loss(
                    losses=entropy, weights=weights)

                tf.contrib.summary.scalar('loss/policy', policy_loss)
                tf.contrib.summary.scalar('loss/value', value_loss)
                tf.contrib.summary.scalar('loss/entropy', entropy_loss)

                tf.contrib.summary.scalar('gradient_norm',
                                          tf.global_norm(grads))
                tf.contrib.summary.scalar('gradient_norm/clipped',
                                          tf.global_norm(grads_clipped))

                tf.contrib.summary.scalar('scale_diag', self.policy.scale_diag)
                tf.contrib.summary.scalar('entropy', entropy_mean)
                tf.contrib.summary.scalar('kl', kl)

    def _eval(self):
        states, actions, rewards, next_states, weights = self.rollout(
            self.inference_strategy, episodes=self.params.episodes)

        episodic_rewards = tf.reduce_mean(tf.reduce_sum(rewards, axis=-1))
        print(self.global_step.numpy(), 'episodic_rewards/eval',
              episodic_rewards.numpy())

        with tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar('episodic_rewards/eval',
                                      episodic_rewards)

            if self.include_histograms:
                for i in range(self.env.observation_space.shape[0]):
                    tf.contrib.summary.histogram('states/{}/eval'.format(i),
                                                 states[..., i])
                for i in range(self.env.action_space.shape[0]):
                    tf.contrib.summary.histogram('actions/{}/eval'.format(i),
                                                 actions[..., i])
                tf.contrib.summary.histogram('rewards/eval', rewards)

        # save checkpoint
        checkpoint_prefix = os.path.join(self.job_dir, 'ckpt')
        self.checkpoint.save(file_prefix=checkpoint_prefix)

    def train(self):
        for it in range(self.params.train_iters):
            # training
            self._train()

            # evaluation
            if it % self.params.eval_interval == self.params.eval_interval - 1:
                self._eval()
