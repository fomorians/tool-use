import os
import gc
import sys
import gym
import tensorflow as tf

import pyoneer as pynr
import pyoneer.rl as pyrl

from tool_use.timer import Timer
from tool_use.models import Model
from tool_use.parallel_rollout import ParallelRollout
from tool_use.wrappers import ObservationCoordinates, ObservationNormalization

from tensorflow.python.keras.utils import losses_utils


class Trainer:
    def __init__(self, job_dir, params):
        self.job_dir = job_dir
        self.params = params

        # environment
        def env_constructor(env_name):
            def _constructor():
                env = gym.make(env_name)
                env = ObservationCoordinates(env)
                env = ObservationNormalization(env)
                return env

            return _constructor

        # TODO: use ray
        self.env = pyrl.envs.BatchEnv(
            constructor=env_constructor(self.params.env_name),
            batch_size=os.cpu_count(),
            blocking=False,
        )
        self.env_perceptual = pyrl.envs.BatchEnv(
            constructor=env_constructor("PerceptualTrapTube-v0"),
            batch_size=os.cpu_count(),
            blocking=False,
        )
        self.env_structural = pyrl.envs.BatchEnv(
            constructor=env_constructor("StructuralTrapTube-v0"),
            batch_size=os.cpu_count(),
            blocking=False,
        )
        self.env_symbolic = pyrl.envs.BatchEnv(
            constructor=env_constructor("SymbolicTrapTube-v0"),
            batch_size=os.cpu_count(),
            blocking=False,
        )

        # seeding
        self.env.seed(self.params.seed)
        self.env_perceptual.seed(self.params.seed)
        self.env_structural.seed(self.params.seed)
        self.env_symbolic.seed(self.params.seed)

        # optimization
        self.optimizer = tf.optimizers.Adam(learning_rate=self.params.learning_rate)

        # models
        self.model = Model(action_space=self.env.action_space)

        # strategies
        self.exploration_strategy = pyrl.strategies.SampleStrategy(
            self.model.get_distribution
        )
        self.inference_strategy = pyrl.strategies.ModeStrategy(
            self.model.get_distribution
        )

        # normalization
        self.rewards_moments = pynr.nn.ExponentialMovingMoments(
            shape=(), rate=self.params.reward_decay
        )

        # checkpoints
        self.checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer,
            model=self.model,
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
        # TODO: use ray
        self.rollout = ParallelRollout(
            self.env, max_episode_steps=self.params.max_episode_steps
        )
        self.rollout_perceptual = ParallelRollout(
            self.env_perceptual, max_episode_steps=self.params.max_episode_steps
        )
        self.rollout_structural = ParallelRollout(
            self.env_structural, max_episode_steps=self.params.max_episode_steps
        )
        self.rollout_symbolic = ParallelRollout(
            self.env_symbolic, max_episode_steps=self.params.max_episode_steps
        )

    @tf.function
    def _train(self, transitions):
        observations, actions, rewards, observations_next, weights = transitions
        episodic_rewards = tf.reduce_mean(tf.reduce_sum(rewards, axis=-1))

        tf.print("episodic_rewards/train", episodic_rewards)
        tf.debugging.assert_less_equal(
            episodic_rewards, 1.0, message="episodic rewards must equal <= 1"
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

        dist_anchor, values = self.model(observations, training=False, reset_state=True)
        log_probs_anchor = dist_anchor.log_prob(actions)

        # TODO: convert to classes
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

        # TODO: tf-2-alpha fails here
        # tf.summary.histogram("actions", actions[..., 0], step=self.optimizer.iterations)
        # tf.summary.histogram(
        #     "directions", actions[..., 1], step=self.optimizer.iterations
        # )

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
            dataset = dataset.batch(self.params.batch_size, drop_remainder=True)
            dataset = dataset.repeat(self.params.epochs)

            # prefetch to gpu if available
            if tf.test.is_gpu_available():
                dataset = dataset.apply(
                    tf.data.experimental.prefetch_to_device("/gpu:0")
                )
            else:
                dataset = dataset.prefetch(self.params.episodes_train)

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
                dist, values = self.model(observations, training=True, reset_state=True)
                log_probs = dist.log_prob(actions)
                entropy = dist.entropy()

                value_loss_fn = tf.losses.MeanSquaredError()

                # losses
                # TODO: convert to classes
                policy_loss = pyrl.losses.clipped_policy_gradient_loss(
                    log_probs=log_probs,
                    log_probs_anchor=log_probs_anchor,
                    advantages=advantages,
                    epsilon_clipping=self.params.epsilon_clipping,
                    sample_weight=weights,
                )
                value_loss = value_loss_fn(
                    y_pred=values[..., None],
                    y_true=returns[..., None],
                    sample_weight=weights * self.params.value_coef,
                )
                # TODO: convert to classes
                entropy_loss = -losses_utils.compute_weighted_loss(
                    losses=entropy, sample_weight=weights * self.params.entropy_coef
                )
                regularization_loss = tf.add_n(
                    [
                        tf.nn.l2_loss(tvar) * self.params.l2_coef
                        for tvar in self.model.trainable_variables
                    ]
                )
                loss = tf.add_n(
                    [policy_loss, value_loss, entropy_loss, regularization_loss]
                )

            # compute gradients
            grads = tape.gradient(loss, self.model.trainable_variables)
            if self.params.grad_clipping is not None:
                grads_clipped, _ = tf.clip_by_global_norm(
                    grads, self.params.grad_clipping
                )
            else:
                grads_clipped = grads

            grads_and_vars = zip(grads_clipped, self.model.trainable_variables)

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
                "gradient_norm",
                tf.linalg.global_norm(grads),
                step=self.optimizer.iterations,
            )
            tf.summary.scalar(
                "gradient_norm/clipped",
                tf.linalg.global_norm(grads_clipped),
                step=self.optimizer.iterations,
            )

            tf.summary.scalar("entropy", entropy_mean, step=self.optimizer.iterations)

    def _eval(self, rollout_fn, name):
        observations, actions, rewards, observations_next, weights = rollout_fn(
            self.inference_strategy, episodes=self.params.episodes_eval
        )
        episodic_rewards = tf.reduce_mean(tf.reduce_sum(rewards, axis=-1))

        tf.print("episodic_rewards/eval/{}".format(name), episodic_rewards)
        tf.debugging.assert_less_equal(
            episodic_rewards, 1.0, message="episodic rewards must equal <= 1"
        )
        tf.summary.scalar(
            "episodic_rewards/eval/{}".format(name),
            episodic_rewards,
            step=self.optimizer.iterations,
        )

    def train(self):
        for it in range(self.params.train_iters):
            tf.print("iteration:", it)

            # training
            with Timer() as train_timer:
                with tf.device("/cpu:0"):
                    transitions = self.rollout(
                        self.exploration_strategy, episodes=self.params.episodes_train
                    )

                self._train(transitions)

            tf.summary.scalar(
                "time/train", train_timer.duration, step=self.optimizer.iterations
            )

            # save checkpoint
            checkpoint_prefix = os.path.join(self.job_dir, "ckpt")
            self.checkpoint.save(file_prefix=checkpoint_prefix)

            # evaluation
            with Timer() as eval_timer:
                with tf.device("/cpu:0"):
                    self._eval(self.rollout, self.params.env_name)
                    self._eval(self.rollout_perceptual, "PerceptualTrapTube-v0")
                    self._eval(self.rollout_structural, "StructuralTrapTube-v0")
                    self._eval(self.rollout_symbolic, "SymbolicTrapTube-v0")

            tf.summary.scalar(
                "time/eval", eval_timer.duration, step=self.optimizer.iterations
            )

            # flush each iteration
            sys.stdout.flush()
            sys.stderr.flush()
            self.summary_writer.flush()

            gc.collect()
