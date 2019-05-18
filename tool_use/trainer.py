import sys
import tensorflow as tf

import pyoneer as pynr
import pyoneer.rl as pyrl

from tool_use.env import create_env
from tool_use.data import create_dataset
from tool_use.model import Model
from tool_use.distribute import parallel_rollout

from tensorflow.python.keras.utils import losses_utils


class Trainer:
    def __init__(self, job_dir, params):
        self.job_dir = job_dir
        self.params = params

        # environment
        self.env = create_env(self.params.env_name, self.params.seed)

        # optimization
        self.optimizer = tf.optimizers.Adam(learning_rate=self.params.learning_rate)

        # models
        self.model = Model(action_space=self.env.action_space)

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

    def _batch_train(self, batch):
        with tf.GradientTape() as tape:
            # forward passes
            log_probs, entropy, values = self.model.get_training_output(
                batch["observations"],
                batch["actions_prev"],
                batch["rewards_prev"],
                batch["actions"],
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

    @tf.function
    def _train(self, transitions):
        episodic_rewards = tf.reduce_mean(
            tf.reduce_sum(transitions["rewards"], axis=-1)
        )

        tf.print("episodic_rewards/train", episodic_rewards)
        tf.debugging.assert_less_equal(
            episodic_rewards, 1.0, message="episodic rewards must equal <= 1"
        )

        self.rewards_moments(
            transitions["rewards"], sample_weight=transitions["weights"], training=True
        )

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

        dist_anchor, values = self.model.get_training_output(
            transitions["observations"],
            transitions["actions_prev"],
            transitions["rewards_prev"],
            transitions["actions"],
            training=False,
            reset_state=True,
        )
        log_probs_anchor = dist_anchor.log_prob(transitions["actions"])

        advantages = self.advantages_fn(
            rewards=rewards_norm, values=values, sample_weight=transitions["weights"]
        )
        returns = self.returns_fn(
            rewards=rewards_norm, sample_weight=transitions["weights"]
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
        tf.summary.histogram(
            "actions", transitions["actions"][..., 0], step=self.optimizer.iterations
        )
        tf.summary.histogram(
            "directions", transitions["actions"][..., 1], step=self.optimizer.iterations
        )

        # dataset creation
        data = {
            "observations": transitions["observations"],
            "actions": transitions["actions"],
            "actions_prev": transitions["actions_prev"],
            "observations_next": transitions["observations_next"],
            "rewards_norm": rewards_norm,
            "rewards_prev": transitions["rewards_prev"],
            "advantages": advantages,
            "returns": returns,
            "log_probs_anchor": log_probs_anchor,
            "weights": transitions["weights"],
        }
        dataset = create_dataset(data)

        # training loop
        for batch in dataset:
            self._batch_train(batch)

    def _eval(self, env_name):
        transitions = parallel_rollout(
            model=self.model,
            env_name=env_name,
            max_episode_steps=self.params.max_episode_steps,
            episodes=self.params.episodes_eval,
            seed=self.params.seed,
            training=False,
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
                transitions = parallel_rollout(
                    model=self.model,
                    env_name=self.params.env_name,
                    max_episode_steps=self.params.max_episode_steps,
                    episodes=self.params.episodes_train,
                    seed=self.params.seed + it,
                    training=True,
                )
                self._train(transitions)

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

            tf.summary.scalar(
                "time/eval", eval_stopwatch.duration, step=self.optimizer.iterations
            )

            # flush each iteration
            sys.stdout.flush()
            sys.stderr.flush()
            self.summary_writer.flush()
