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


def compute_sample_probs(priorities, alpha=0.7):
    priorities_exp = tf.math.pow(priorities, alpha)
    probs = priorities_exp / tf.math.reduce_sum(priorities_exp)
    probs = tf.math.reduce_sum(probs, axis=-1)  # aggregate over time
    return probs


def compute_is_weight(probs, buffer_size, beta=0.5):
    is_weight = tf.pow(buffer_size * probs, -beta)
    is_weight /= tf.math.reduce_max(is_weight)
    return is_weight


class Algorithm:
    def __init__(self, job_dir, params, eval_env_names):
        self.params = params
        self.eval_env_names = eval_env_names

        # seed manager
        self.seeds = Seeds(
            base_seed=params.seed,
            env_batch_size=params.batch_size,
            eval_env_names=eval_env_names,
        )

        self.proto_env = create_env(params.env_name)()
        self.actor = Actor(self.proto_env.action_space, params.batch_size)
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
            critic = Critic(params.batch_size)
            critic_target = Critic(params.batch_size)
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

    def _compute_is_weight(self, priorities):
        probs = compute_sample_probs(priorities, alpha=self.params.alpha)
        is_weight = compute_is_weight(
            probs, buffer_size=self.buffer.size, beta=self.params.beta
        )
        return is_weight[:, None]

    def _train_critic(self, batch):
        targets = self._compute_targets(batch)

        inputs = {"observations": batch["observations"], "actions": batch["actions"]}
        is_weight = self._compute_is_weight(batch["priorities"])
        action_values_list = []

        for i, (critic, critic_optimizer) in enumerate(
            zip(self.critics, self.critic_optimizers)
        ):
            # compute loss
            with tf.GradientTape() as tape:
                action_values = critic(inputs, training=True, reset_state=True)
                losses = self.critic_loss_fn(
                    y_pred=action_values[..., None],
                    y_true=targets[..., None],
                    sample_weight=batch["weights"][..., None] * is_weight[..., None],
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

            action_values_list.append(action_values)

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

        action_values_min = tf.reduce_min(action_values_list, axis=0)
        td_errors = targets - action_values_min
        return td_errors

    def _train_actor(self, batch):
        is_weight = self._compute_is_weight(batch["priorities"])

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
                sample_weight=batch["weights"] * is_weight,
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

            for i, action_name in enumerate(["move", "grasp"]):
                for j in range(4):
                    tf.summary.histogram(
                        "actions/data/{}/{}".format(action_name, j),
                        tf.boolean_mask(batch["actions"][..., i, j], mask),
                        step=self.actor_optimizer.iterations,
                    )
                    tf.summary.histogram(
                        "actions/sampled/{}/{}".format(action_name, j),
                        tf.boolean_mask(actions[..., i, j], mask),
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
        is_weight = self._compute_is_weight(batch["priorities"])

        # compute loss
        with tf.GradientTape() as tape:
            losses = self.temperature_loss_fn(
                log_probs=log_probs,
                log_alpha=self.log_alpha,
                sample_weight=batch["weights"] * is_weight,
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

    @tf.function
    def _train_batch(self, batch):
        td_errors = self._train_critic(batch)
        log_probs = self._train_actor(batch)
        self._train_alpha(batch, log_probs)

        # update target network
        for critic, critic_target in zip(self.critics, self.critic_targets):
            pynr.variables.update_variables(
                critic.variables,
                critic_target.variables,
                rate=self.params.target_update_rate,
            )

        return td_errors

    def _train_data(self, transitions, indices):
        dataset = create_dataset(
            (transitions, indices), batch_size=self.params.batch_size
        )
        for batch, indices in dataset:
            td_errors = self._train_batch(batch)

            # update priorities
            batch["priorities"] = tf.abs(td_errors)
            self.buffer.update(indices, batch)

    def _train(self, it):
        if self.params.episodes_train > 0:
            # collect new transitions
            transitions = collect_transitions(
                env_name=self.params.env_name,
                episodes=self.params.episodes_train,
                batch_size=self.params.batch_size,
                policy=self.explore_policy,
                seed=self.seeds.train_seed(it),
            )

            # initialize priorities to 1M (arbitrary)
            transitions["priorities"] = np.ones_like(transitions["weights"]) * 1e6

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

        probs = compute_sample_probs(
            self.buffer["priorities"], alpha=self.params.alpha
        ).numpy()
        transitions, indices = self.buffer.sample(
            size=self.params.episodes_train_sample, p=probs, return_indices=True
        )
        self._train_data(transitions, indices)

    def _eval(self):
        if self.params.episodes_eval == 0:
            return

        for env_name in constants.env_names:
            # collect new transitions
            transitions = collect_transitions(
                env_name=env_name,
                episodes=self.params.episodes_eval,
                batch_size=self.params.batch_size,
                policy=self.exploit_policy,
                seed=self.seeds.eval_seed(env_name),
            )

            # compute summaries
            with self.job.summary_context("eval"):
                episodic_reward = tf.reduce_mean(
                    tf.reduce_sum(transitions["rewards"], axis=-1)
                )

                tf.summary.scalar(
                    "episodic_rewards/eval/{}".format(env_name),
                    episodic_reward,
                    step=self.actor_optimizer.iterations,
                )

    def _initialize_models(self):
        # create mock inputs to initialize models
        mock_observation = self.proto_env.observation_space.sample()
        mock_action = self.proto_env.action_space.sample()

        batch_size = self.params.batch_size
        max_episode_steps = self.proto_env.spec.max_episode_steps

        mock_observations = np.asarray(mock_observation)[None, None, ...]
        mock_actions = np.asarray(mock_action)[None, None, ...]

        mock_observations = np.tile(
            mock_observations, [batch_size, max_episode_steps, 1, 1, 1]
        )
        mock_actions = np.tile(mock_actions, [batch_size, max_episode_steps, 1, 1])

        critic_inputs = {"observations": mock_observations, "actions": mock_actions}
        actor_inputs = {"observations": mock_observations, "actions_prev": mock_actions}

        self.actor(actor_inputs, training=True, reset_state=True)

        for critic, critic_target in zip(self.critics, self.critic_targets):
            # initialize models so we can update target the variables
            critic(critic_inputs, training=True, reset_state=True)
            critic_target(critic_inputs, training=True, reset_state=True)

            # update target network by hard-copying all variables
            pynr.variables.update_variables(critic.variables, critic_target.variables)

    def _train_iter(self, it):
        # train
        tf.print("train", it)
        with pynr.debugging.Stopwatch() as train_stopwatch:
            self._train(it)

        with self.job.summary_context("train"):
            tf.summary.scalar(
                "time/train",
                train_stopwatch.duration,
                step=self.actor_optimizer.iterations,
            )

        # eval
        if it % self.params.eval_interval == 0:
            tf.print("eval", it)
            with pynr.debugging.Stopwatch() as eval_stopwatch:
                self._eval()

            with self.job.summary_context("eval"):
                tf.summary.scalar(
                    "time/eval",
                    eval_stopwatch.duration,
                    step=self.actor_optimizer.iterations,
                )

        self.job.save(checkpoint_number=it)
        self.job.flush_summaries()

    def train(self):
        self._initialize_models()

        # sample random transitions to pre-fill the transitions buffer
        if self.params.episodes_init > 0:
            transitions = collect_transitions(
                env_name=self.params.env_name,
                episodes=self.params.episodes_init,
                batch_size=self.params.batch_size,
                policy=self.explore_policy,
                seed=self.seeds.train_seed(0),
            )

            # initialize priorities to 1M (arbitrary)
            transitions["priorities"] = np.ones_like(transitions["weights"]) * 1e6

            self.buffer.append(transitions)

        # begin training iterations
        for it in range(1, self.params.train_iters + 1):
            self._train_iter(it)

            # flush each iteration
            sys.stdout.flush()
            sys.stderr.flush()
