import pyoneer as pynr
import tensorflow as tf
import tensorflow_probability as tfp

from tool_use.layers import ResidualBlock


class Actor(tf.keras.Model):
    def __init__(self, action_space, **kwargs):
        super(Actor, self).__init__(**kwargs)

        self.action_space = action_space

        kernel_initializer = tf.initializers.VarianceScaling(scale=2.0)
        logits_initializer = tf.initializers.VarianceScaling(scale=1.0)

        self.initial_hidden_state = tf.Variable(tf.zeros(shape=[1, 64]), trainable=True)
        self.hidden_state = None

        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=2,
            strides=1,
            padding="same",
            activation=pynr.activations.swish,
            kernel_initializer=kernel_initializer,
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=2,
            strides=1,
            padding="same",
            activation=pynr.activations.swish,
            kernel_initializer=kernel_initializer,
        )

        self.downsample1 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
        self.residual_block1 = ResidualBlock(filters=64)
        self.downsample2 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
        self.residual_block2 = ResidualBlock(filters=64)

        self.global_pool = tf.keras.layers.GlobalMaxPool2D()

        self.dense_hidden = tf.keras.layers.Dense(
            units=64,
            activation=pynr.activations.swish,
            kernel_initializer=kernel_initializer,
        )

        self.rnn = tf.keras.layers.GRU(
            units=64, return_sequences=True, return_state=True
        )

        self.move_logits = tf.keras.layers.Dense(
            units=action_space.shape[-1],
            activation=None,
            kernel_initializer=logits_initializer,
        )
        self.grasp_logits = tf.keras.layers.Dense(
            units=action_space.shape[-1],
            activation=None,
            kernel_initializer=logits_initializer,
        )

    def call(self, inputs, training=None, reset_state=None):
        observations = inputs["observations"]
        actions_prev = inputs["actions_prev"]

        batch_size, steps, height, width, channels = observations.shape
        action_size, action_dim = actions_prev.shape[2:]

        observations = tf.reshape(
            observations, [batch_size * steps, height, width, channels]
        )
        actions_prev = tf.reshape(
            actions_prev, [batch_size * steps, action_size * action_dim]
        )

        hidden = self.conv1(observations)
        hidden = self.conv2(hidden)

        hidden = self.downsample1(hidden)
        hidden = self.residual_block1(hidden)
        hidden = self.downsample2(hidden)
        hidden = self.residual_block2(hidden)

        hidden = self.global_pool(hidden)

        hidden = self.dense_hidden(tf.concat([hidden, actions_prev], axis=-1))

        hidden = tf.reshape(hidden, [batch_size, steps, self.dense_hidden.units])

        if self.hidden_state is None or reset_state:
            self.hidden_state = tf.tile(self.initial_hidden_state, [batch_size, 1])

        hidden, self.hidden_state = self.rnn(hidden, initial_state=self.hidden_state)

        move_logits = self.move_logits(hidden)
        grasp_logits = self.grasp_logits(hidden)

        return move_logits, grasp_logits

    def get_dist(self, *args, **kwargs):
        move_logits, grasp_logits = self.call(*args, **kwargs)
        move_dist = tfp.distributions.RelaxedOneHotCategorical(
            logits=move_logits, temperature=1.0
        )
        grasp_dist = tfp.distributions.RelaxedOneHotCategorical(
            logits=grasp_logits, temperature=1.0
        )
        return move_dist, grasp_dist

    def explore(self, *args, **kwargs):
        move_dist, grasp_dist = self.get_dist(*args, **kwargs)
        move_actions = move_dist.sample()
        grasp_actions = grasp_dist.sample()
        actions = tf.stack([move_actions, grasp_actions], axis=-2)
        return actions

    def exploit(self, *args, **kwargs):
        move_logits, grasp_logits = self.call(*args, **kwargs)
        move_actions = tf.math.softmax(move_logits)
        grasp_actions = tf.math.softmax(grasp_logits)
        actions = tf.stack([move_actions, grasp_actions], axis=-2)
        return actions

    def sample(self, *args, **kwargs):
        move_dist, grasp_dist = self.get_dist(*args, **kwargs)
        move_actions = move_dist.sample()
        grasp_actions = grasp_dist.sample()
        move_log_probs = move_dist.log_prob(move_actions)
        grasp_log_probs = grasp_dist.log_prob(grasp_actions)
        actions = tf.stack([move_actions, grasp_actions], axis=-2)
        log_probs = tf.add_n([move_log_probs, grasp_log_probs])
        return actions, log_probs
