import sys

import gym
import numpy as np
import pyoneer as pynr
import pyoneer.rl as pyrl
import tensorflow as tf

from tool_use import constants
from tool_use.seeds import Seeds
from tool_use.utils import create_env, collect_transitions, create_dataset
from tool_use.sac.actor import Actor
from tool_use.sac.critic import Critic


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

        env = create_env(params.env_name)()
        self.actor = Actor(env.action_space)
        self.actor_optimizer = tf.optimizers.Adam(learning_rate=params.learning_rate)

        self.log_alpha = tf.Variable(0.0, trainable=True)
        self.alpha_optimizer = tf.optimizers.Adam(learning_rate=params.learning_rate)

        checkpointables = {
            "actor": self.actor,
            "actor_optimizer": self.actor_optimizer,
            "log_alpha": self.log_alpha,
            "alpha_optimizer": self.alpha_optimizer,
        }

        self.critics = []
        self.critic_targets = []
        self.critic_optimizers = []

        for i in range(params.num_critics):
            critic = Critic()
            critic_target = Critic()
            critic_optimizer = tf.optimizers.Adam(learning_rate=params.learning_rate)

            self.critics.append(critic)
            self.critic_targets.append(critic_target)
            self.critic_optimizers.append(critic_optimizer)

            checkpointables["critic{}".format(i)] = critic
            checkpointables["critic{}_target".format(i)] = critic_target
            checkpointables["critic{}_optimizer".format(i)] = critic_target

        self.job = pynr.jobs.Job(directory=job_dir, **checkpointables)

        self.explore_policy = getattr(self.actor, "explore")
        self.exploit_policy = getattr(self.actor, "exploit")

        self.buffer = pyrl.transitions.TransitionBuffer(max_size=params.episodes_max)

        self.critic_loss_fn = tf.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE
        )
        self.actor_loss_fn = pyrl.losses.SoftPolicyGradient(
            reduction=tf.keras.losses.Reduction.NONE
        )
        self.temperature_loss_fn = pyrl.losses.SoftPolicyEntropy(
            reduction=tf.keras.losses.Reduction.NONE,
            target_entropy=params.target_entropy,
        )

    @property
    def alpha(self):
        return tf.exp(self.log_alpha)

    def _compute_targets(self, batch):
        actor_inputs = {
            "observations": batch["observations_next"],
            "actions_prev": batch["actions"],
            "rewards_prev": batch["rewards"],
        }
        actions_next, log_probs_next = self.actor.sample(
            actor_inputs, training=False, reset_state=True
        )

        critic_inputs = {
            "observations": batch["observations_next"],
            "actions": actions_next,
        }
        action_values_next = [
            critic(critic_inputs, training=False, reset_state=True)
            for critic in self.critic_targets
        ]
        action_values_next_min = tf.reduce_min(action_values_next, axis=0)

        values_next = action_values_next_min - self.alpha * log_probs_next

        targets = tf.where(
            batch["dones"],
            batch["rewards"],
            batch["rewards"] + self.params.discount_factor * values_next,
        )
        return tf.stop_gradient(targets)

    def _train_critic(self, batch):
        targets = self._compute_targets(batch)

        inputs = {"observations": batch["observations"], "actions": batch["actions"]}

        for i, (critic, critic_optimizer) in enumerate(
            zip(self.critics, self.critic_optimizers)
        ):
            # compute loss
            with tf.GradientTape() as tape:
                action_values = critic(inputs, training=True, reset_state=True)
                losses = self.critic_loss_fn(
                    y_pred=action_values[..., None],
                    y_true=targets[..., None],
                    sample_weight=batch["weights"][..., None],
                )
                loss = pynr.losses.compute_weighted_loss(
                    losses, sample_weight=batch["weights"]
                )
                regularization_loss = tf.add_n(
                    [
                        self.params.l2_scale * tf.nn.l2_loss(tvar)
                        for tvar in critic.trainable_variables
                    ]
                )
                total_loss = loss + regularization_loss

            # optimize gradients
            grads = tape.gradient(total_loss, critic.trainable_variables)
            grads_clipped, _ = tf.clip_by_global_norm(grads, self.params.grad_clipping)
            grads_and_vars = zip(grads_clipped, critic.trainable_variables)
            critic_optimizer.apply_gradients(grads_and_vars)

            # compute summaries
            with self.job.summary_context("train"):
                mask = batch["weights"] > 0

                grad_norm = tf.linalg.global_norm(grads)
                grad_norm_clipped = tf.linalg.global_norm(grads_clipped)

                tf.summary.histogram(
                    "targets/critic/{}".format(i),
                    tf.boolean_mask(targets, mask),
                    step=critic_optimizer.iterations,
                )
                tf.summary.histogram(
                    "action_values/critic/{}".format(i),
                    tf.boolean_mask(action_values, mask),
                    step=critic_optimizer.iterations,
                )
                tf.summary.scalar(
                    "losses/critic/{}".format(i), loss, step=critic_optimizer.iterations
                )
                tf.summary.scalar(
                    "losses/critic/{}/regularization".format(i),
                    regularization_loss,
                    step=critic_optimizer.iterations,
                )
                tf.summary.scalar(
                    "grad_norm/critic/{}".format(i),
                    grad_norm,
                    step=critic_optimizer.iterations,
                )
                tf.summary.scalar(
                    "grad_norm_clipped/critic/{}".format(i),
                    grad_norm_clipped,
                    step=critic_optimizer.iterations,
                )

    def _train_actor(self, batch):
        # compute loss
        with tf.GradientTape() as tape:
            actor_inputs = {
                "observations": batch["observations"],
                "actions_prev": batch["actions_prev"],
                "rewards_prev": batch["rewards_prev"],
            }
            actions, log_probs = self.actor.sample(
                actor_inputs, training=True, reset_state=True
            )

            critic_inputs = {"observations": batch["observations"], "actions": actions}
            action_values = [
                critic(critic_inputs, training=False, reset_state=True)
                for critic in self.critics
            ]
            action_values_min = tf.reduce_min(action_values, axis=0)

            losses = self.actor_loss_fn(
                log_probs=log_probs,
                action_values=action_values_min,
                alpha=self.alpha,
                sample_weight=batch["weights"],
            )
            loss = pynr.losses.compute_weighted_loss(
                losses, sample_weight=batch["weights"]
            )
            regularization_loss = tf.add_n(
                [
                    self.params.l2_scale * tf.nn.l2_loss(tvar)
                    for tvar in self.actor.trainable_variables
                ]
            )
            total_loss = loss + regularization_loss

        # optimize gradients
        grads = tape.gradient(total_loss, self.actor.trainable_variables)
        grads_clipped, _ = tf.clip_by_global_norm(grads, self.params.grad_clipping)
        grads_and_vars = zip(grads_clipped, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(grads_and_vars)

        # compute summaries
        with self.job.summary_context("train"):
            mask = batch["weights"] > 0

            grad_norm = tf.linalg.global_norm(grads)
            grad_norm_clipped = tf.linalg.global_norm(grads_clipped)

            tf.summary.histogram(
                "log_probs",
                tf.boolean_mask(log_probs, mask),
                step=self.actor_optimizer.iterations,
            )
            tf.summary.scalar(
                "losses/actor", loss, step=self.actor_optimizer.iterations
            )
            tf.summary.scalar(
                "losses/actor/regularization",
                regularization_loss,
                step=self.actor_optimizer.iterations,
            )
            tf.summary.scalar(
                "grad_norm/actor", grad_norm, step=self.actor_optimizer.iterations
            )
            tf.summary.scalar(
                "grad_norm_clipped/actor",
                grad_norm_clipped,
                step=self.actor_optimizer.iterations,
            )

        return log_probs

    def _train_alpha(self, batch, log_probs):
        # compute loss
        with tf.GradientTape() as tape:
            losses = self.temperature_loss_fn(
                log_probs=log_probs,
                log_alpha=self.log_alpha,
                sample_weight=batch["weights"],
            )
            loss = pynr.losses.compute_weighted_loss(
                losses, sample_weight=batch["weights"]
            )

        # optimize gradients
        trainable_variables = [self.log_alpha]
        grads = tape.gradient(loss, trainable_variables)
        grads_and_vars = zip(grads, trainable_variables)
        self.alpha_optimizer.apply_gradients(grads_and_vars)

        # compute summaries
        with self.job.summary_context("train"):
            grad_norm = tf.linalg.global_norm(grads)
            log_probs_mean = pynr.losses.compute_weighted_loss(
                log_probs, sample_weight=batch["weights"]
            )

            tf.summary.scalar("alpha", self.alpha, step=self.alpha_optimizer.iterations)
            tf.summary.scalar(
                "log_probs/mean", log_probs_mean, step=self.alpha_optimizer.iterations
            )
            tf.summary.scalar(
                "losses/alpha", loss, step=self.alpha_optimizer.iterations
            )
            tf.summary.scalar(
                "grad_norm/alpha", grad_norm, step=self.alpha_optimizer.iterations
            )

    # @tf.function
    def _train_data(self, transitions):
        dataset = (
            tf.data.Dataset.from_tensor_slices(transitions)
            .batch(self.params.batch_size)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
        for batch in dataset:
            self._train_critic(batch)
            log_probs = self._train_actor(batch)
            self._train_alpha(batch, log_probs)

            # update target network
            for critic, critic_target in zip(self.critics, self.critic_targets):
                pynr.variables.update_variables(
                    critic.variables,
                    critic_target.variables,
                    rate=self.params.target_update_rate,
                )

    def _train(self, it):
        if self.params.episodes_train > 0:
            # collect new transitions
            transitions = collect_transitions(
                env_name=self.params.env_name,
                episodes=self.params.episodes_train,
                batch_size=self.params.env_batch_size,
                policy=self.explore_policy,
                seed=self.seeds.train_seed(it),
            )
            self.buffer.append(transitions)

            # compute summaries
            with self.job.summary_context("train"):
                episodic_reward = tf.reduce_mean(
                    tf.reduce_sum(transitions["rewards"], axis=-1)
                )

                tf.summary.scalar(
                    "episodic_rewards/train",
                    episodic_reward,
                    step=self.actor_optimizer.iterations,
                )
                tf.summary.scalar(
                    "buffer/size",
                    self.buffer.size,
                    step=self.actor_optimizer.iterations,
                )

        transitions = self.buffer.sample(size=self.params.episodes_train)
        self._train_data(transitions)

    def _eval(self):
        if self.params.episodes_eval == 0:
            return

        for env_name in constants.env_names:
            # collect new transitions
            transitions = collect_transitions(
                env_name=env_name,
                episodes=self.params.episodes_eval,
                batch_size=self.params.env_batch_size,
                policy=self.exploit_policy,
                seed=self.seeds.eval_seed(env_name),
            )

            # compute summaries
            with self.job.summary_context("eval"):
                episodic_reward = tf.reduce_mean(
                    tf.reduce_sum(transitions["rewards"], axis=-1)
                )

                tf.summary.scalar(
                    "episodic_rewards/eval",
                    episodic_reward,
                    step=self.actor_optimizer.iterations,
                )

    def _initialize_models(self):
        env = create_env(self.params.env_name)()

        # create mock inputs to initialize models
        mock_observation = env.observation_space.sample()
        mock_action = env.action_space.sample()

        mock_observations = np.asarray(mock_observation)[None, None, ...]
        mock_actions = np.asarray(mock_action)[None, None, ...]

        inputs = {"observations": mock_observations, "actions": mock_actions}

        for critic, critic_target in zip(self.critics, self.critic_targets):
            # initialize models so we can update target the variables
            critic(inputs, training=True, reset_state=True)
            critic_target(inputs, training=True, reset_state=True)

            # update target network by hard-copying all variables
            pynr.variables.update_variables(critic.variables, critic_target.variables)

    def _train_iter(self, it):
        # train
        with pynr.debugging.Stopwatch() as train_stopwatch:
            self._train(it)

        with self.job.summary_context("train"):
            tf.summary.scalar(
                "time/train",
                train_stopwatch.duration,
                step=self.actor_optimizer.iterations,
            )

        # eval
        with pynr.debugging.Stopwatch() as eval_stopwatch:
            self._eval()

        with self.job.summary_context("eval"):
            tf.summary.scalar(
                "time/eval",
                eval_stopwatch.duration,
                step=self.actor_optimizer.iterations,
            )

        self.job.save(checkpoint_number=it)

    def train(self):
        self._initialize_models()

        # sample random transitions to pre-fill the transitions buffer
        transitions = collect_transitions(
            env_name=self.params.env_name,
            episodes=self.params.episodes_init,
            batch_size=self.params.env_batch_size,
            policy=self.explore_policy,
            seed=self.seeds.train_seed(0),
        )
        self.buffer.append(transitions)

        # begin training iterations
        for it in range(1, 1 + self.params.train_iters):
            tf.print("iteration:", it - 1)
            self._train_iter(it)

        # flush each iteration
        sys.stdout.flush()
        sys.stderr.flush()
        self.job.flush_summaries()
