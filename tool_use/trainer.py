import sys
import tensorflow as tf

import pyoneer as pynr
import pyoneer.rl as pyrl

from tool_use.env import create_env
from tool_use.data import create_dataset
from tool_use.model import Model
from tool_use.batch_rollout import BatchRollout

from tensorflow.python.keras.utils import losses_utils


class Trainer:
    def __init__(self, job_dir, params):
        self.job_dir = job_dir
        self.params = params

        # optimization
        self.optimizer = tf.optimizers.Adam(learning_rate=self.params.learning_rate)

        # models
        env = create_env(self.params.env_name, self.params.seed)
        self.model = Model(action_space=env.action_space)

        # policies
        self.exploration_policy = pyrl.strategies.Sample(self.model)
        self.inference_policy = pyrl.strategies.Mode(self.model)

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
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint, directory=self.job_dir, max_to_keep=None
        )
        if self.checkpoint_manager.latest_checkpoint:
            status = self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            status.assert_consumed()

        # summaries
        self.summary_writer = tf.summary.create_file_writer(
            self.job_dir, max_queue=100, flush_millis=5 * 60 * 1000
        )
        self.summary_writer.set_as_default()

        # losses
        self.value_loss_fn = tf.losses.MeanSquaredError()
        self.policy_loss_fn = pyrl.losses.ClippedPolicyGradient(
            epsilon_clipping=self.params.epsilon_clipping
        )
        self.entropy_loss_fn = pyrl.losses.PolicyEntropy()

        # targets
        self.advantages_fn = pyrl.targets.GeneralizedAdvantages(
            discount_factor=self.params.discount_factor,
            lambda_factor=self.params.lambda_factor,
            normalize=self.params.normalize_advantages,
        )
        self.returns_fn = pyrl.targets.DiscountedRewards(
            discount_factor=self.params.discount_factor
        )

    def _collect_transitions(self, env_name, episodes, policy, seed):
        with tf.device("/cpu:0"):
            env = pyrl.wrappers.Batch(
                lambda: create_env(env_name, seed),
                batch_size=self.params.env_batch_size,
            )
            rollout = BatchRollout(env, self.params.max_episode_steps)
            transitions = rollout(policy, episodes)
        return transitions

    def _batch_train(self, batch):
        with tf.GradientTape() as tape:
            # forward passes
            log_probs, entropy, values = self.model.get_training_output(
                observations=batch["observations"],
                actions_prev=batch["actions_prev"],
                rewards_prev=batch["rewards_prev"],
                actions=batch["actions"],
                training=True,
                reset_state=True,
            )

            # losses
            policy_loss = self.policy_loss_fn(
                log_probs=log_probs,
                log_probs_anchor=batch["log_probs_anchor"],
                advantages=batch["advantages"],
                sample_weight=batch["weights"],
            )
            value_loss = self.value_loss_fn(
                y_pred=values[..., None],
                y_true=batch["returns"][..., None],
                sample_weight=batch["weights"] * self.params.value_coef,
            )
            entropy_loss = self.entropy_loss_fn(
                entropy=entropy,
                sample_weight=batch["weights"] * self.params.entropy_coef,
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
            grads_clipped, _ = tf.clip_by_global_norm(grads, self.params.grad_clipping)
        else:
            grads_clipped = grads
        grads_and_vars = zip(grads_clipped, self.model.trainable_variables)

        # optimization
        self.optimizer.apply_gradients(grads_and_vars)

        # summaries
        entropy_mean = losses_utils.compute_weighted_loss(
            losses=entropy, sample_weight=batch["weights"]
        )
        gradient_norm = tf.linalg.global_norm(grads)
        clipped_gradient_norm = tf.linalg.global_norm(grads_clipped)

        tf.summary.scalar("loss/policy", policy_loss, step=self.optimizer.iterations)
        tf.summary.scalar("loss/value", value_loss, step=self.optimizer.iterations)
        tf.summary.scalar("loss/entropy", entropy_loss, step=self.optimizer.iterations)
        tf.summary.scalar(
            "loss/regularization", regularization_loss, step=self.optimizer.iterations
        )
        tf.summary.scalar(
            "gradient_norm", gradient_norm, step=self.optimizer.iterations
        )
        tf.summary.scalar(
            "gradient_norm/clipped",
            clipped_gradient_norm,
            step=self.optimizer.iterations,
        )
        tf.summary.scalar("entropy", entropy_mean, step=self.optimizer.iterations)

    def _train(self, transitions):
        episodic_rewards = tf.reduce_mean(
            tf.reduce_sum(transitions["rewards"], axis=-1)
        )
        tf.print("episodic_rewards/train", episodic_rewards)
        tf.debugging.assert_less_equal(
            episodic_rewards, 1.0, message="episodic rewards must equal <= 1"
        )
        tf.summary.scalar(
            "episodic_rewards/train", episodic_rewards, step=self.optimizer.iterations
        )

        # update reward moments
        self.rewards_moments(
            transitions["rewards"], sample_weight=transitions["weights"], training=True
        )

        # normalize rewards
        if self.params.center_reward:
            rewards_norm = pynr.math.normalize(
                transitions["rewards"],
                loc=self.rewards_moments.mean,
                scale=self.rewards_moments.std,
                sample_weight=transitions["weights"],
            )
        else:
            rewards_norm = pynr.math.safe_divide(
                transitions["rewards"], self.rewards_moments.std
            )

        # compute anchor values
        log_probs_anchor, _, values_anchor = self.model.get_training_output(
            observations=transitions["observations"],
            actions_prev=transitions["actions_prev"],
            rewards_prev=transitions["rewards_prev"],
            actions=transitions["actions"],
            training=False,
            reset_state=True,
        )

        # targets
        advantages = self.advantages_fn(
            rewards=rewards_norm,
            values=values_anchor,
            sample_weight=transitions["weights"],
        )
        returns = self.returns_fn(
            rewards=rewards_norm, sample_weight=transitions["weights"]
        )

        # summaries
        tf.summary.scalar(
            "rewards/mean", self.rewards_moments.mean, step=self.optimizer.iterations
        )
        tf.summary.scalar(
            "rewards/std", self.rewards_moments.std, step=self.optimizer.iterations
        )
        tf.summary.histogram(
            "actions", transitions["actions"][..., 0], step=self.optimizer.iterations
        )
        tf.summary.histogram(
            "directions", transitions["actions"][..., 1], step=self.optimizer.iterations
        )

        data = dict(transitions)
        data["rewards_norm"] = rewards_norm
        data["log_probs_anchor"] = log_probs_anchor
        data["advantages"] = advantages
        data["returns"] = returns

        # dataset
        dataset = create_dataset(
            data, batch_size=self.params.batch_size, epochs=self.params.epochs
        )

        # training
        for batch in dataset:
            self._batch_train(batch)

    def _eval(self, env_name):
        transitions = self._collect_transitions(
            env_name, self.params.episodes_eval, self.inference_policy, self.params.seed
        )
        episodic_rewards = tf.reduce_mean(
            tf.reduce_sum(transitions["rewards"], axis=-1)
        )

        tf.print("episodic_rewards/eval/{}".format(env_name), episodic_rewards)
        tf.debugging.assert_less_equal(
            episodic_rewards, 1.0, message="episodic rewards must equal <= 1"
        )
        tf.summary.scalar(
            "episodic_rewards/eval/{}".format(env_name),
            episodic_rewards,
            step=self.optimizer.iterations,
        )

    def train(self):
        for it in range(self.params.train_iters):
            tf.print("iteration:", it)

            # training
            with pynr.debugging.Stopwatch() as train_stopwatch:
                transitions = self._collect_transitions(
                    self.params.env_name,
                    self.params.episodes_train,
                    self.exploration_policy,
                    self.params.seed + it,
                )
                self._train(transitions)

            tf.print("time/train", train_stopwatch.duration)
            tf.summary.scalar(
                "time/train", train_stopwatch.duration, step=self.optimizer.iterations
            )

            # save checkpoint
            self.checkpoint_manager.save()

            # evaluation
            with pynr.debugging.Stopwatch() as eval_stopwatch:
                with tf.device("/cpu:0"):
                    self._eval(self.params.env_name)
                    self._eval("PerceptualTrapTube-v0")
                    self._eval("StructuralTrapTube-v0")
                    self._eval("SymbolicTrapTube-v0")

            tf.print("time/eval", eval_stopwatch.duration)
            tf.summary.scalar(
                "time/eval", eval_stopwatch.duration, step=self.optimizer.iterations
            )

            # flush each iteration
            sys.stdout.flush()
            sys.stderr.flush()
            self.summary_writer.flush()
