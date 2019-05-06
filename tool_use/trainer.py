import os
import sys
import tensorflow as tf

import pyoneer as pynr
import pyoneer.rl as pyrl

import gym_tool_use

from tool_use.timer import Timer
from tool_use.models import PolicyModel, ValueModel
from tool_use.parallel_rollout import ParallelRollout

from tensorflow.python.keras.utils import losses_utils


class Trainer:
    def __init__(self, job_dir, params):
        self.job_dir = job_dir
        self.params = params

        # environment
        def env_constructor():
            env = gym_tool_use.BridgeBuildingEnv(
                observation_type="rgb",
                random_colors=False,
                randomly_swap_water_and_boxes=False,
            )
            return env

        num_envs = os.cpu_count()
        self.env = pyrl.envs.BatchEnv(
            constructor=env_constructor, batch_size=num_envs, blocking=False
        )

        # seeding
        self.env.seed(self.params.seed)

        # optimization
        self.global_step = tf.Variable(0, dtype=tf.int64)
        self.optimizer = tf.optimizers.Adam(learning_rate=self.params.learning_rate)

        # models
        self.policy_model = PolicyModel(action_size=self.env.action_space.n)
        self.value_model = ValueModel()

        # strategies
        self.exploration_strategy = pyrl.strategies.SampleStrategy(self.policy_model)
        self.inference_strategy = pyrl.strategies.ModeStrategy(self.policy_model)

        # normalization
        self.rewards_moments = pynr.nn.ExponentialMovingMoments(
            shape=(), rate=self.params.reward_decay
        )

        # checkpoints
        self.checkpoint = tf.train.Checkpoint(
            global_step=self.global_step,
            optimizer=self.optimizer,
            policy_model=self.policy_model,
            value_model=self.value_model,
            rewards_moments=self.rewards_moments,
        )
        checkpoint_path = tf.train.latest_checkpoint(self.job_dir)
        if checkpoint_path is not None:
            self.checkpoint.restore(checkpoint_path)

        # summaries
        self.summary_writer = tf.summary.create_file_writer(
            self.job_dir, max_queue=100, flush_millis=5 * 60 * 1000
        )
        self.summary_writer.set_as_default()

        # rollouts
        self.rollout = ParallelRollout(
            self.env, max_episode_steps=self.params.max_episode_steps
        )

    def _train(self):
        # compute data
        # force rollouts to cpu
        with tf.device("/cpu:0"):
            (observations, actions, rewards, observations_next, weights) = self.rollout(
                self.exploration_strategy, episodes=self.params.episodes
            )

        self.rewards_moments(rewards, weights=weights, training=True)

        if self.params.center_reward:
            rewards_norm = pynr.math.normalize(
                rewards,
                loc=self.rewards_moments.mean,
                scale=self.rewards_moments.std,
                weights=weights,
            )
        else:
            rewards_norm = pynr.math.safe_divide(rewards, self.rewards_moments.std)

        dist_anchor = self.policy_model(observations, training=False)
        log_probs_anchor = dist_anchor.log_prob(actions)
        values = self.value_model(observations, training=False)

        advantages = pyrl.targets.generalized_advantages(
            rewards=rewards_norm,
            values=values,
            discount_factor=self.params.discount_factor,
            lambda_factor=self.params.lambda_factor,
            weights=weights,
            normalize=self.params.normalize_advantages,
        )
        returns = pyrl.targets.discounted_rewards(
            rewards=rewards_norm,
            discount_factor=self.params.discount_factor,
            weights=weights,
        )

        episodic_rewards = tf.reduce_mean(tf.reduce_sum(rewards, axis=-1))
        print(
            self.optimizer.iterations.numpy(),
            "episodic_rewards/train",
            episodic_rewards.numpy(),
        )

        # summaries
        tf.summary.scalar(
            "episodic_rewards/train", episodic_rewards, step=self.optimizer.iterations
        )
        tf.summary.scalar(
            "rewards/mean", self.rewards_moments.mean, step=self.optimizer.iterations
        )
        tf.summary.scalar(
            "rewards/std", self.rewards_moments.std, step=self.optimizer.iterations
        )

        with tf.device("/cpu:0"):
            dataset = tf.data.Dataset.from_tensor_slices(
                (
                    observations,
                    actions,
                    observations_next,
                    rewards_norm,
                    advantages,
                    returns,
                    log_probs_anchor,
                    weights,
                )
            )
            dataset = dataset.batch(self.params.batch_size)
            dataset = dataset.repeat(self.params.epochs)

            # prefetch to gpu if available
            if tf.test.is_gpu_available():
                dataset = dataset.apply(
                    tf.data.experimental.prefetch_to_device("/gpu:0")
                )
            else:
                dataset = dataset.prefetch(self.params.episodes)

        trainable_variables = (
            self.policy_model.trainable_variables + self.value_model.trainable_variables
        )

        for (
            observations,
            actions,
            observations_next,
            rewards_norm,
            advantages,
            returns,
            log_probs_anchor,
            weights,
        ) in dataset:
            with tf.GradientTape() as tape:
                # forward passes
                dist = self.policy_model(observations, training=True)
                log_probs = dist.log_prob(actions)
                values = self.value_model(observations, training=True)
                entropy = dist.entropy()

                value_loss_fn = tf.losses.MeanSquaredError()

                import ipdb

                ipdb.set_trace()

                # losses
                policy_loss = pyrl.losses.clipped_policy_gradient_loss(
                    log_probs=log_probs,
                    log_probs_anchor=log_probs_anchor,
                    advantages=advantages,
                    epsilon_clipping=self.params.epsilon_clipping,
                    sample_weight=weights,
                )
                value_loss = value_loss_fn(
                    y_pred=values,
                    y_true=returns,
                    sample_weight=weights * self.params.value_coef,
                )
                entropy_loss = -losses_utils.compute_weighted_loss(
                    losses=entropy, sample_weight=weights * self.params.entropy_coef
                )
                regularization_loss = tf.add_n(
                    [
                        tf.nn.l2_loss(tvar) * self.params.l2_coef
                        for tvar in trainable_variables
                    ]
                )
                loss = tf.add_n(
                    [policy_loss, value_loss, entropy_loss, regularization_loss]
                )

            # compute gradients
            grads = tape.gradient(loss, trainable_variables)
            if self.params.grad_clipping is not None:
                grads_clipped, _ = tf.clip_by_global_norm(
                    grads, self.params.grad_clipping
                )
            grads_and_vars = zip(grads_clipped, trainable_variables)

            # optimization
            self.optimizer.apply_gradients(grads_and_vars)

            # summaries
            entropy_mean = losses_utils.compute_weighted_loss(
                losses=entropy, sample_weight=weights
            )

            tf.summary.scalar(
                "loss/policy", policy_loss, step=self.optimizer.iterations
            )
            tf.summary.scalar("loss/value", value_loss, step=self.optimizer.iterations)
            tf.summary.scalar(
                "loss/entropy", entropy_loss, step=self.optimizer.iterations
            )
            tf.summary.scalar(
                "loss/regularization",
                regularization_loss,
                step=self.optimizer.iterations,
            )

            tf.summary.scalar(
                "gradient_norm", tf.global_norm(grads), step=self.optimizer.iterations
            )
            tf.summary.scalar(
                "gradient_norm/clipped",
                tf.global_norm(grads_clipped),
                step=self.optimizer.iterations,
            )

            tf.summary.scalar("entropy", entropy_mean, step=self.optimizer.iterations)

    def _eval(self):
        with tf.device("/cpu:0"):
            (observations, actions, rewards, observations_next, weights) = self.rollout(
                self.inference_strategy, episodes=self.params.episodes
            )

        episodic_rewards = tf.reduce_mean(tf.reduce_sum(rewards, axis=-1))
        print(
            self.global_step.numpy(), "episodic_rewards/eval", episodic_rewards.numpy()
        )

        tf.summary.scalar(
            "episodic_rewards/eval", episodic_rewards, step=self.optimizer.iterations
        )

        # save checkpoint
        checkpoint_prefix = os.path.join(self.job_dir, "ckpt")
        self.checkpoint.save(file_prefix=checkpoint_prefix)

    def train(self):
        for it in range(self.params.train_iters):
            # training
            with Timer() as train_timer:
                self._train()

            tf.summary.scalar(
                "time/train", train_timer.duration, step=self.optimizer.iterations
            )

            # evaluation
            if it % self.params.eval_interval == self.params.eval_interval - 1:
                with Timer() as eval_timer:
                    self._eval()

                tf.summary.scalar(
                    "time/eval", eval_timer.duration, step=self.optimizer.iterations
                )

            # flush each iteration
            sys.stdout.flush()
            self.summary_writer.flush()
