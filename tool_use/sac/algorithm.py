import sys

import gym
import numpy as np
import pyoneer as pynr
import pyoneer.rl as pyrl
import tensorflow as tf

from tool_use.sac.models import Policy, QFunction


class Algorithm:
    def __init__(self, job_dir, params):
        self.params = params

        self.env = pyrl.wrappers.Batch(
            create_env(params.env), batch_size=params.batch_size
        )
        self.env.seed(params.seed)

        self.policy = Policy(self.env.action_space)
        self.policy_optimizer = tf.optimizers.Adam(learning_rate=params.learning_rate)

        self.log_alpha = tf.Variable(0.0, trainable=True)
        self.alpha_optimizer = tf.optimizers.Adam(learning_rate=params.learning_rate)

        checkpointables = {
            "directory": job_dir,
            "policy": self.policy,
            "policy_optimizer": self.policy_optimizer,
            "log_alpha": self.log_alpha,
            "alpha_optimizer": self.alpha_optimizer,
        }

        self.q_fns = []
        self.q_fn_targets = []
        self.q_fn_optimizers = []

        for i in range(params.num_q_fns):
            q_fn = QFunction()
            q_fn_target = QFunction()
            q_fn_optimizer = tf.optimizers.Adam(learning_rate=params.learning_rate)

            self.q_fns.append(q_fn)
            self.q_fn_targets.append(q_fn_target)
            self.q_fn_optimizers.append(q_fn_optimizer)

            checkpointables["q_fn{}".format(i)] = q_fn
            checkpointables["q_fn{}_target".format(i)] = q_fn_target
            checkpointables["q_fn{}_optimizer".format(i)] = q_fn_target

        self.job = pynr.jobs.Job(**checkpointables)

        self.agent = Agent(self.policy)

        self.explore_strat = FnStrategy(self.agent, "explore")
        self.exploit_strat = FnStrategy(self.agent, "exploit")

        self.rollout = pynr.rl.rollouts.Rollout(self.env)
        self.buffer = pyrl.transitions.TransitionBuffer(max_size=params.max_size)

        self.q_loss_fn = tf.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE
        )
        self.policy_loss_fn = pyrl.losses.SoftPolicyGradient(
            reduction=tf.keras.losses.Reduction.NONE
        )
        self.alpha_loss_fn = pyrl.losses.SoftPolicyEntropy(
            reduction=tf.keras.losses.Reduction.NONE,
            target_entropy=params.target_entropy,
        )

    @property
    def alpha(self):
        return tf.exp(self.log_alpha)

    def _compute_targets(self, batch):
        policy = self.policy(
            batch["observations_next"], training=False, reset_state=True
        )
        actions_next = policy.sample()
        log_probs_next = policy.log_prob(actions_next)

        inputs = {"observations": batch["observations_next"], "actions": actions_next}

        action_values_next = [
            q_fn(inputs, training=False, reset_state=True) for q_fn in self.q_fn_targets
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

        for i, (q_fn, q_fn_optimizer) in enumerate(
            zip(self.q_fns, self.q_fn_optimizers)
        ):
            # compute loss
            with tf.GradientTape() as tape:
                action_values = q_fn(inputs, training=True, reset_state=True)
                losses = self.q_loss_fn(
                    y_pred=action_values[..., None],
                    y_true=targets[..., None],
                    sample_weight=batch["weights"][..., None],
                )
                loss = pynr.losses.compute_weighted_loss(
                    losses, sample_weight=batch["weights"]
                )

            # optimize gradients
            grads = tape.gradient(loss, q_fn.trainable_variables)
            grads_clipped, _ = tf.clip_by_global_norm(grads, self.params.grad_clipping)
            grads_and_vars = zip(grads_clipped, q_fn.trainable_variables)
            q_fn_optimizer.apply_gradients(grads_and_vars)

            # compute summaries
            with self.job.summary_context("train"):
                mask = batch["weights"] > 0

                grad_norm = tf.linalg.global_norm(grads)
                grad_norm_clipped = tf.linalg.global_norm(grads_clipped)

                tf.summary.histogram(
                    "targets/q_fn/{}".format(i),
                    tf.boolean_mask(targets, mask),
                    step=q_fn_optimizer.iterations,
                )
                tf.summary.histogram(
                    "action_values/q_fn/{}".format(i),
                    tf.boolean_mask(action_values, mask),
                    step=q_fn_optimizer.iterations,
                )
                tf.summary.scalar(
                    "losses/q_fn/{}".format(i), loss, step=q_fn_optimizer.iterations
                )
                tf.summary.scalar(
                    "grad_norm/q_fn/{}".format(i),
                    grad_norm,
                    step=q_fn_optimizer.iterations,
                )
                tf.summary.scalar(
                    "grad_norm_clipped/q_fn/{}".format(i),
                    grad_norm_clipped,
                    step=q_fn_optimizer.iterations,
                )

    def _train_actor(self, batch):
        # compute loss
        with tf.GradientTape() as tape:
            policy = self.policy(batch["observations"], training=True, reset_state=True)
            actions = policy.sample()
            log_probs = policy.log_prob(actions)

            inputs = {"observations": batch["observations"], "actions": actions}

            action_values = [
                q_fn(inputs, training=False, reset_state=True) for q_fn in self.q_fns
            ]
            action_values_min = tf.reduce_min(action_values, axis=0)

            losses = self.policy_loss_fn(
                log_probs=log_probs,
                action_values=action_values_min,
                alpha=self.alpha,
                sample_weight=batch["weights"],
            )
            loss = pynr.losses.compute_weighted_loss(
                losses, sample_weight=batch["weights"]
            )

        # optimize gradients
        grads = tape.gradient(loss, self.policy.trainable_variables)
        grads_clipped, _ = tf.clip_by_global_norm(grads, self.params.grad_clipping)
        grads_and_vars = zip(grads_clipped, self.policy.trainable_variables)
        self.policy_optimizer.apply_gradients(grads_and_vars)

        # compute summaries
        with self.job.summary_context("train"):
            mask = batch["weights"] > 0

            grad_norm = tf.linalg.global_norm(grads)
            grad_norm_clipped = tf.linalg.global_norm(grads_clipped)

            for i in range(self.policy.num_outputs):
                tf.summary.histogram(
                    "actions/data/{}".format(i),
                    tf.boolean_mask(batch["actions"][..., i], mask),
                    step=self.policy_optimizer.iterations,
                )
                tf.summary.histogram(
                    "actions/sampled/{}".format(i),
                    tf.boolean_mask(actions[..., i], mask),
                    step=self.policy_optimizer.iterations,
                )
            tf.summary.histogram(
                "log_probs",
                tf.boolean_mask(log_probs, mask),
                step=self.policy_optimizer.iterations,
            )
            tf.summary.scalar(
                "losses/policy", loss, step=self.policy_optimizer.iterations
            )
            tf.summary.scalar(
                "grad_norm/policy", grad_norm, step=self.policy_optimizer.iterations
            )
            tf.summary.scalar(
                "grad_norm_clipped/policy",
                grad_norm_clipped,
                step=self.policy_optimizer.iterations,
            )

        return log_probs

    def _train_alpha(self, batch, log_probs):
        # compute loss
        with tf.GradientTape() as tape:
            losses = self.alpha_loss_fn(
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
            for q_fn, q_fn_target in zip(self.q_fns, self.q_fn_targets):
                pynr.variables.update_variables(
                    q_fn.variables,
                    q_fn_target.variables,
                    rate=self.params.target_update_rate,
                )

    def _train(self):
        if self.params.episodes_train > 0:
            # collect new transitions
            transitions = self.rollout(
                self.explore_strat, episodes=self.params.episodes_train
            )
            if self.params.flatten:
                transitions_flat = flatten_transitions(transitions)
                self.buffer.append(transitions_flat)
            else:
                self.buffer.append(transitions)

            # compute summaries
            with self.job.summary_context("train"):
                episodic_reward = episodic_mean(transitions["rewards"])

                tf.summary.scalar(
                    "episodic_rewards/train",
                    episodic_reward,
                    step=self.policy_optimizer.iterations,
                )
                tf.summary.scalar(
                    "buffer/size",
                    self.buffer.size,
                    step=self.policy_optimizer.iterations,
                )

        transitions = self.buffer.sample(size=self.params.num_samples)
        self._train_data(transitions)

    def _eval(self):
        if self.params.episodes_eval > 0:
            # collect new transitions
            transitions = self.rollout(
                self.exploit_strat, episodes=self.params.episodes_eval
            )

            # compute summaries
            with self.job.summary_context("eval"):
                episodic_reward = episodic_mean(transitions["rewards"])

                tf.summary.scalar(
                    "episodic_rewards/eval",
                    episodic_reward,
                    step=self.policy_optimizer.iterations,
                )

    def _initialize_models(self):
        # create mock inputs to initialize models
        mock_observation = self.env.observation_space.sample()
        mock_action = self.env.action_space.sample()

        mock_observations = np.asarray(mock_observation)[None, None, ...]
        mock_actions = np.asarray(mock_action)[None, None, ...]

        inputs = {"observations": mock_observations, "actions": mock_actions}

        for q_fn, q_fn_target in zip(self.q_fns, self.q_fn_targets):
            # initialize models so we can update target the variables
            q_fn(inputs, training=True, reset_state=True)
            q_fn_target(inputs, training=True, reset_state=True)

            # update target network by hard-copying all variables
            pynr.variables.update_variables(q_fn.variables, q_fn_target.variables)

    def _train_iter(self, it):
        tf.print("iteration:", it)

        # train
        with pynr.debugging.Stopwatch() as train_stopwatch:
            self._train()

        with self.job.summary_context("train"):
            tf.summary.scalar(
                "time/train",
                train_stopwatch.duration,
                step=self.policy_optimizer.iterations,
            )

        # eval
        with pynr.debugging.Stopwatch() as eval_stopwatch:
            self._eval()

        with self.job.summary_context("eval"):
            tf.summary.scalar(
                "time/eval",
                eval_stopwatch.duration,
                step=self.policy_optimizer.iterations,
            )

        self.job.save(checkpoint_number=it)

    def train(self):
        self._initialize_models()

        # sample random transitions to pre-fill the transitions buffer
        if self.params.steps_init > 0:
            while self.buffer.size < self.params.steps_init:
                transitions = self.rollout(self.explore_strat, episodes=1)
                if self.params.flatten:
                    transitions_flat = flatten_transitions(transitions)
                    self.buffer.append(transitions_flat)
                else:
                    self.buffer.append(transitions)
                print(
                    "Filling buffer... ({:.2f}%)".format(
                        (self.buffer.size / self.params.steps_init) * 100
                    )
                )

        # begin training iterations
        for it in range(self.params.iterations):
            self._train_iter(it)

        # flush each iteration
        sys.stdout.flush()
        sys.stderr.flush()
        self.job.flush_summaries()
