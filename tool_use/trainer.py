import os
import gym
import sys
import time
import tensorflow as tf

import pyoneer as pynr
import pyoneer.rl as pyrl

from tool_use.models import Model
from tool_use.wrappers import RangeNormalize
from tool_use.parallel_rollout import ParallelRollout


class Trainer:
    def __init__(self, job_dir, params):
        self.job_dir = job_dir
        self.params = params

        # environment
        def env_constructor():
            env = gym.make(self.params.env)
            env = RangeNormalize(env)
            return env

        num_envs = os.cpu_count()
        self.env = pyrl.envs.BatchEnv(
            constructor=env_constructor, batch_size=num_envs, blocking=False)

        observation_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.shape[0]

        # seeding
        self.env.seed(self.params.seed)

        # optimization
        self.global_step = tf.train.create_global_step()
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.params.learning_rate)

        # models
        self.model = Model(action_size=action_size, scale=self.params.scale)

        # strategies
        self.exploration_strategy = pyrl.strategies.SampleStrategy(self.model)
        self.inference_strategy = pyrl.strategies.ModeStrategy(self.model)

        # normalization
        self.rewards_moments = pynr.nn.ExponentialMovingMoments(
            shape=(), rate=self.params.reward_decay)

        # checkpoints
        self.checkpoint = tf.train.Checkpoint(
            global_step=self.global_step,
            optimizer=self.optimizer,
            model=self.model,
            rewards_moments=self.rewards_moments)
        checkpoint_path = tf.train.latest_checkpoint(self.job_dir)
        if checkpoint_path is not None:
            self.checkpoint.restore(checkpoint_path)

        # summaries
        self.summary_writer = tf.contrib.summary.create_file_writer(
            self.job_dir, max_queue=100, flush_millis=5 * 60 * 1000)
        self.summary_writer.set_as_default()

        # rollouts
        self.rollout = ParallelRollout(
            self.env, max_episode_steps=self.env.spec.max_episode_steps)

    def _train(self):
        # compute data
        # force rollouts to cpu
        with tf.device('/cpu:0'):
            (observations, actions, rewards, observations_next,
             weights) = self.rollout(
                 self.exploration_strategy, episodes=self.params.episodes)

        mini_episodes = (
            (self.params.episodes * self.env.spec.max_episode_steps) //
            self.params.horizon)

        observations = observations.reshape(mini_episodes, self.params.horizon,
                                            observations.shape[-1])
        actions = actions.reshape(mini_episodes, self.params.horizon,
                                  actions.shape[-1])
        rewards = rewards.reshape(mini_episodes, self.params.horizon)
        observations_next = observations_next.reshape(
            mini_episodes, self.params.horizon, observations_next.shape[-1])
        weights = weights.reshape(mini_episodes, self.params.horizon)

        self.rewards_moments(rewards, weights=weights, training=True)

        if self.params.center_reward:
            rewards_norm = pynr.math.normalize(
                rewards,
                loc=self.rewards_moments.mean,
                scale=self.rewards_moments.std,
                weights=weights)
        else:
            rewards_norm = pynr.math.safe_divide(rewards,
                                                 self.rewards_moments.std)

        predictions = self.model.forward(
            observations, actions, include=['log_probs', 'values'])
        values = predictions['values']
        log_probs_anchor = predictions['log_probs']

        advantages = pyrl.targets.generalized_advantages(
            rewards=rewards_norm,
            values=predictions['values'],
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

        # summaries
        with tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar('episodic_rewards/train',
                                      episodic_rewards)
            tf.contrib.summary.scalar('rewards/mean',
                                      self.rewards_moments.mean)
            tf.contrib.summary.scalar('rewards/std', self.rewards_moments.std)

        with tf.device('/cpu:0'):
            dataset = tf.data.Dataset.from_tensor_slices(
                (observations, actions, observations_next, rewards_norm,
                 advantages, returns, log_probs_anchor, weights))
            dataset = dataset.batch(self.params.batch_size)
            dataset = dataset.repeat(self.params.epochs)

            # prefetch to gpu if available
            if tf.test.is_gpu_available():
                dataset = dataset.apply(
                    tf.data.experimental.prefetch_to_device('/gpu:0'))
            else:
                dataset = dataset.prefetch(mini_episodes)

        for (observations, actions, observations_next, rewards_norm,
             advantages, returns, log_probs_anchor, weights) in dataset:
            with tf.GradientTape() as tape:
                # forward passes
                predictions = self.model.forward(
                    observations, actions, observations_next, training=True)
                log_probs = predictions['log_probs']
                entropy = predictions['entropy']
                values = predictions['values']

                forward_preds = predictions['observations_next_embedding_pred']
                forward_labels = predictions['observations_next_embedding']

                inverse_preds = predictions['actions_embedding_pred']
                inverse_labels = predictions['actions_embedding']

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
                forward_loss = tf.losses.mean_squared_error(
                    predictions=forward_preds,
                    labels=forward_labels,
                    weights=weights[..., None] * self.params.forward_coef)
                inverse_loss = tf.losses.mean_squared_error(
                    predictions=inverse_preds,
                    labels=inverse_labels,
                    weights=weights[..., None] * self.params.inverse_coef)
                regularization_loss = tf.add_n([
                    tf.nn.l2_loss(tvar) * self.params.l2_coef
                    for tvar in self.model.trainable_variables
                ])
                loss = tf.add_n([
                    policy_loss, value_loss, entropy_loss, forward_loss,
                    inverse_loss, regularization_loss
                ])

            # compute gradients
            grads = tape.gradient(loss, self.model.trainable_variables)
            if self.params.grad_clipping is not None:
                grads_clipped, _ = tf.clip_by_global_norm(
                    grads, self.params.grad_clipping)
            grads_and_vars = zip(grads_clipped, self.model.trainable_variables)

            # optimization
            self.optimizer.apply_gradients(
                grads_and_vars, global_step=self.global_step)

            # summaries
            with tf.contrib.summary.always_record_summaries():
                entropy_mean = tf.losses.compute_weighted_loss(
                    losses=predictions['entropy'], weights=weights)

                tf.contrib.summary.scalar('loss/policy', policy_loss)
                tf.contrib.summary.scalar('loss/value', value_loss)
                tf.contrib.summary.scalar('loss/entropy', entropy_loss)
                tf.contrib.summary.scalar('loss/forward', forward_loss)
                tf.contrib.summary.scalar('loss/inverse', inverse_loss)
                tf.contrib.summary.scalar('loss/regularization',
                                          regularization_loss)

                tf.contrib.summary.scalar('gradient_norm',
                                          tf.global_norm(grads))
                tf.contrib.summary.scalar('gradient_norm/clipped',
                                          tf.global_norm(grads_clipped))

                tf.contrib.summary.scalar('scale_diag',
                                          self.model.policy.scale_diag)
                tf.contrib.summary.scalar('entropy', entropy_mean)

    def _eval(self):
        with tf.device('/cpu:0'):
            (observations, actions, rewards, observations_next,
             weights) = self.rollout(
                 self.inference_strategy, episodes=self.params.episodes)

        episodic_rewards = tf.reduce_mean(tf.reduce_sum(rewards, axis=-1))
        print(self.global_step.numpy(), 'episodic_rewards/eval',
              episodic_rewards.numpy())

        with tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar('episodic_rewards/eval',
                                      episodic_rewards)

        # save checkpoint
        checkpoint_prefix = os.path.join(self.job_dir, 'ckpt')
        self.checkpoint.save(file_prefix=checkpoint_prefix)

    def train(self):
        for it in range(self.params.train_iters):
            start_time = time.time()

            # training
            self._train()

            end_time = time.time()
            train_time = end_time - start_time

            # evaluation
            if it % self.params.eval_interval == self.params.eval_interval - 1:
                start_time = time.time()

                self._eval()

                end_time = time.time()
                eval_time = end_time - start_time

                with tf.contrib.summary.always_record_summaries():
                    tf.contrib.summary.scalar('time/eval', eval_time)

            with tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar('time/train', train_time)

            # flush each iteration
            sys.stdout.flush()
            self.summary_writer.flush()
