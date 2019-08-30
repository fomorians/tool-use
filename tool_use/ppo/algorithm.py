import sys
import tensorflow as tf

import pyoneer as pynr
import pyoneer.rl as pyrl

from tool_use.seeds import Seeds
from tool_use.utils import create_env, collect_transitions, create_dataset
from tool_use.ppo.policy import Policy


class Algorithm:
    def __init__(self, job_dir, params, eval_env_names):
        self.params = params
        self.eval_env_names = eval_env_names

        # seed manager
        self.seeds = Seeds(
            base_seed=params.seed,
            env_batch_size=params.env_batch_size,
            eval_env_names=eval_env_names,
        )

        # optimization
        self.optimizer = tf.optimizers.Adam(learning_rate=self.params.learning_rate)

        # model
        env = create_env(params.env_name)()
        self.policy = Policy(action_space=env.action_space, use_l2rl=params.use_l2rl)

        # policies
        self.exploration_policy = pyrl.strategies.Sample(self.policy)
        self.inference_policy = pyrl.strategies.Mode(self.policy)

        # normalization
        self.extrinsic_rewards_moments = pynr.moments.ExponentialMovingMoments(
            shape=(), rate=self.params.reward_decay
        )
        self.intrinsic_rewards_moments = pynr.moments.ExponentialMovingMoments(
            shape=(), rate=self.params.reward_decay
        )

        # checkpoints
        self.checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer,
            policy=self.policy,
            extrinsic_rewards_moments=self.extrinsic_rewards_moments,
            intrinsic_rewards_moments=self.intrinsic_rewards_moments,
        )
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint, directory=job_dir, max_to_keep=None
        )
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(
                self.checkpoint_manager.latest_checkpoint
            ).expect_partial()

        # summaries
        self.summary_writer = tf.summary.create_file_writer(
            job_dir, max_queue=100, flush_millis=5 * 60 * 1000
        )
        self.summary_writer.set_as_default()

        # losses
        self.value_loss_fn = tf.losses.MeanSquaredError()
        self.policy_loss_fn = pyrl.losses.ClippedPolicyGradient(
            epsilon_clipping=self.params.epsilon_clipping
        )
        self.entropy_loss_fn = pyrl.losses.PolicyEntropy()
        self.forward_loss_fn = tf.losses.MeanSquaredError()
        self.inverse_move_loss_fn = tf.losses.CategoricalCrossentropy()
        self.inverse_grasp_loss_fn = tf.losses.CategoricalCrossentropy()

        # targets
        self.advantages_fn = pyrl.targets.GeneralizedAdvantages(
            discount_factor=self.params.discount_factor,
            lambda_factor=self.params.lambda_factor,
            normalize=self.params.normalize_advantages,
        )
        self.returns_fn = pyrl.targets.DiscountedReturns(
            discount_factor=self.params.discount_factor
        )

    def _compute_loss(
        self, policy_loss, value_loss, entropy_loss, intrinsic_loss, regularization_loss
    ):
        if self.params.use_icm:
            return tf.add_n(
                [
                    policy_loss,
                    value_loss,
                    entropy_loss,
                    intrinsic_loss,
                    regularization_loss,
                ]
            )
        else:
            return tf.add_n(
                [policy_loss, value_loss, entropy_loss, regularization_loss]
            )

    def _batch_train(self, batch):
        with tf.GradientTape() as tape:
            # forward passes
            outputs = self.policy.get_training_outputs(
                inputs=batch, training=True, reset_state=True
            )
            log_probs = outputs["log_probs"]
            entropy = outputs["entropy"]
            values = outputs["values"]

            move_hot = tf.one_hot(
                batch["actions"][..., 0], depth=self.policy.action_space.nvec[0]
            )
            grasp_hot = tf.one_hot(
                batch["actions"][..., 1], depth=self.policy.action_space.nvec[1]
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
            forward_loss = self.forward_loss_fn(
                y_pred=outputs["embedding_next_pred"],
                y_true=outputs["embedding_next"],
                sample_weight=batch["weights"] * self.params.forward_coef,
            )
            inverse_move_loss = self.inverse_move_loss_fn(
                y_pred=outputs["move_pred"],
                y_true=move_hot,
                sample_weight=batch["weights"],
            )
            inverse_grasp_loss = self.inverse_grasp_loss_fn(
                y_pred=outputs["grasp_pred"],
                y_true=grasp_hot,
                sample_weight=batch["weights"],
            )
            inverse_loss = (
                inverse_move_loss + inverse_grasp_loss
            ) * self.params.inverse_coef
            intrinsic_loss = (forward_loss + inverse_loss) * self.params.intrinsic_coef
            regularization_loss = tf.add_n(
                [
                    tf.nn.l2_loss(tvar) * self.params.l2_coef
                    for tvar in self.policy.trainable_variables
                ]
            )
            loss = self._compute_loss(
                policy_loss=policy_loss,
                value_loss=value_loss,
                entropy_loss=entropy_loss,
                intrinsic_loss=intrinsic_loss,
                regularization_loss=regularization_loss,
            )

        # compute gradients
        grads = tape.gradient(loss, self.policy.trainable_variables)
        if self.params.grad_clipping is not None:
            grads_clipped, _ = tf.clip_by_global_norm(grads, self.params.grad_clipping)
        else:
            grads_clipped = grads
        grads_and_vars = zip(grads_clipped, self.policy.trainable_variables)

        # optimization
        self.optimizer.apply_gradients(grads_and_vars)

        # summaries
        entropy_mean = pynr.losses.compute_weighted_loss(
            losses=entropy, sample_weight=batch["weights"]
        )
        gradient_norm = tf.linalg.global_norm(grads)
        clipped_gradient_norm = tf.linalg.global_norm(grads_clipped)

        tf.summary.scalar("loss/policy", policy_loss, step=self.optimizer.iterations)
        tf.summary.scalar("loss/value", value_loss, step=self.optimizer.iterations)
        tf.summary.scalar("loss/entropy", entropy_loss, step=self.optimizer.iterations)
        tf.summary.scalar("loss/forward", forward_loss, step=self.optimizer.iterations)
        tf.summary.scalar("loss/inverse", inverse_loss, step=self.optimizer.iterations)
        tf.summary.scalar(
            "loss/inverse/move", inverse_move_loss, step=self.optimizer.iterations
        )
        tf.summary.scalar(
            "loss/inverse/grasp", inverse_grasp_loss, step=self.optimizer.iterations
        )
        tf.summary.scalar(
            "loss/intrinsic", intrinsic_loss, step=self.optimizer.iterations
        )
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
        extrinsic_rewards = transitions["rewards"]

        episodic_rewards = tf.reduce_mean(tf.reduce_sum(extrinsic_rewards, axis=1))
        tf.print("episodic_rewards/train", episodic_rewards)
        tf.debugging.assert_less_equal(
            episodic_rewards, 1.0, message="episodic rewards must equal <= 1"
        )
        tf.summary.scalar(
            "episodic_rewards/train", episodic_rewards, step=self.optimizer.iterations
        )

        # compute anchor values
        outputs = self.policy.get_training_outputs(
            inputs=transitions, training=False, reset_state=True
        )
        log_probs_anchor = outputs["log_probs"]
        values_anchor = outputs["values"]

        intrinsic_rewards = tf.reduce_sum(
            tf.math.squared_difference(
                outputs["embedding_next"], outputs["embedding_next_pred"]
            ),
            axis=-1,
        )
        episodic_intrinsic_rewards = tf.reduce_mean(
            tf.reduce_sum(intrinsic_rewards, axis=-1)
        )
        tf.summary.scalar(
            "episodic_intrinsic_rewards/train",
            episodic_intrinsic_rewards,
            step=self.optimizer.iterations,
        )

        # update reward moments
        self.extrinsic_rewards_moments(
            extrinsic_rewards, sample_weight=transitions["weights"], training=True
        )
        self.intrinsic_rewards_moments(
            intrinsic_rewards, sample_weight=transitions["weights"], training=True
        )

        # normalize rewards
        if self.params.center_reward:
            extrinsic_rewards_norm = pynr.math.normalize(
                extrinsic_rewards,
                loc=self.extrinsic_rewards_moments.mean,
                scale=self.extrinsic_rewards_moments.std,
                sample_weight=transitions["weights"],
            )
            intrinsic_rewards_norm = pynr.math.normalize(
                intrinsic_rewards,
                loc=self.intrinsic_rewards_moments.mean,
                scale=self.intrinsic_rewards_moments.std,
                sample_weight=transitions["weights"] * self.params.intrinsic_scale,
            )
        else:
            extrinsic_rewards_norm = (
                tf.math.divide_no_nan(
                    extrinsic_rewards, self.extrinsic_rewards_moments.std
                )
                * transitions["weights"]
            )
            intrinsic_rewards_norm = tf.math.divide_no_nan(
                intrinsic_rewards, self.intrinsic_rewards_moments.std
            ) * (transitions["weights"] * self.params.intrinsic_scale)

        rewards_norm = extrinsic_rewards_norm + intrinsic_rewards_norm

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
            "extrinsic_rewards/mean",
            self.extrinsic_rewards_moments.mean,
            step=self.optimizer.iterations,
        )
        tf.summary.scalar(
            "extrinsic_rewards/std",
            self.extrinsic_rewards_moments.std,
            step=self.optimizer.iterations,
        )
        tf.summary.scalar(
            "intrinsic_rewards/mean",
            self.intrinsic_rewards_moments.mean,
            step=self.optimizer.iterations,
        )
        tf.summary.scalar(
            "intrinsic_rewards/std",
            self.intrinsic_rewards_moments.std,
            step=self.optimizer.iterations,
        )
        tf.summary.histogram(
            "actions/moves",
            transitions["actions"][..., 0],
            step=self.optimizer.iterations,
        )
        tf.summary.histogram(
            "actions/directions",
            transitions["actions"][..., 1],
            step=self.optimizer.iterations,
        )

        data = dict(transitions)
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
        transitions = collect_transitions(
            env_name=env_name,
            episodes=self.params.episodes_eval,
            batch_size=self.params.env_batch_size,
            policy=self.inference_policy,
            seed=self.seeds.eval_seed(env_name),
        )
        episodic_rewards = tf.reduce_mean(tf.reduce_sum(transitions["rewards"], axis=1))

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
                transitions = collect_transitions(
                    env_name=self.params.env_name,
                    episodes=self.params.episodes_train,
                    batch_size=self.params.env_batch_size,
                    policy=self.exploration_policy,
                    seed=self.seeds.train_seed(it),
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
                for env_name in self.eval_env_names:
                    self._eval(env_name)

            tf.print("time/eval", eval_stopwatch.duration)
            tf.summary.scalar(
                "time/eval", eval_stopwatch.duration, step=self.optimizer.iterations
            )

            # flush each iteration
            sys.stdout.flush()
            sys.stderr.flush()
            self.summary_writer.flush()
