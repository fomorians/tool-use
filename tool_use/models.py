import pyoneer as pynr
import tensorflow as tf
import tensorflow_probability as tfp


class Model(tf.keras.Model):
    def __init__(self, action_space):
        super(Model, self).__init__()

        kernel_initializer = tf.initializers.VarianceScaling(scale=2.0)
        logits_initializer = tf.initializers.VarianceScaling(scale=1.0)

        self.initial_hidden_state = tf.Variable(tf.zeros(shape=[1, 64]), trainable=True)
        self.initial_cell_state = tf.Variable(tf.zeros(shape=[1, 64]), trainable=True)

        self.hidden_state = None
        self.cell_state = None

        self.conv2d_hidden1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=2,
            strides=1,
            padding="valid",
            activation=pynr.nn.swish,
            use_bias=True,
            kernel_initializer=kernel_initializer,
        )
        self.conv2d_hidden2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=2,
            strides=1,
            padding="valid",
            activation=pynr.nn.swish,
            use_bias=True,
            kernel_initializer=kernel_initializer,
        )
        self.flatten = tf.keras.layers.Flatten()

        self.one_hot_actions = pynr.layers.OneHotEncoder(depth=action_space.nvec[0])
        self.one_hot_directions = pynr.layers.OneHotEncoder(depth=action_space.nvec[1])
        self.concat = tf.keras.layers.Concatenate()

        self.dense_hidden = tf.keras.layers.Dense(
            units=64,
            activation=pynr.nn.swish,
            use_bias=True,
            kernel_initializer=kernel_initializer,
        )

        self.rnn = tf.keras.layers.LSTM(
            units=64, return_sequences=True, return_state=True
        )

        self.dense_action_logits = tf.keras.layers.Dense(
            units=action_space.nvec[0],
            activation=None,
            kernel_initializer=logits_initializer,
        )
        self.dense_direction_logits = tf.keras.layers.Dense(
            units=action_space.nvec[1],
            activation=None,
            kernel_initializer=logits_initializer,
        )
        self.dense_values_logits = tf.keras.layers.Dense(
            units=1, activation=None, kernel_initializer=logits_initializer
        )

    def get_embedding(
        self, observations, actions_prev, rewards_prev, training=None, reset_state=None
    ):
        batch_size, steps, height, width, channels = observations.shape

        observations = tf.reshape(observations, [batch_size * steps, height, width, channels])
        actions_prev = tf.reshape(actions_prev, [batch_size * steps, actions_prev.shape[-1]])
        rewards_prev = tf.reshape(rewards_prev, [batch_size * steps, 1])

        hidden = self.conv2d_hidden1(observations)
        hidden = self.conv2d_hidden2(hidden)
        hidden = self.flatten(hidden)

        actions_hot_prev = self.one_hot_actions(actions_prev[..., 0])
        directions_hot_prev = self.one_hot_directions(actions_prev[..., 1])

        hidden = self.concat(
            [hidden, actions_hot_prev, directions_hot_prev, rewards_prev]
        )

        hidden = self.dense_hidden(hidden)
        hidden = tf.reshape(hidden, [batch_size, steps, self.dense_hidden.units])

        if self.hidden_state is None or self.cell_state is None or reset_state:
            self.hidden_state = tf.tile(self.initial_hidden_state, [batch_size, 1])
            self.cell_state = tf.tile(self.initial_cell_state, [batch_size, 1])

        hidden, self.hidden_state, self.cell_state = self.rnn(
            hidden, initial_state=(self.hidden_state, self.cell_state)
        )
        return hidden

    def get_training_output(self, observations, actions_prev, rewards_prev, actions, training=None, reset_state=None):
        hidden = self.get_embedding(
            observations,
            actions_prev,
            rewards_prev,
            training=training,
            reset_state=reset_state,
        )

        action_logits = self.dense_action_logits(hidden)
        direction_logits = self.dense_direction_logits(hidden)
        values_logits = self.dense_values_logits(hidden)

        action_dist = tfp.distributions.Categorical(logits=action_logits)
        direction_dist = tfp.distributions.Categorical(logits=direction_logits)
        dist = pynr.distributions.MultiCategorical([action_dist, direction_dist])

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = values_logits[..., 0]

        return log_probs, entropy, values

    def call(
        self, observations, actions_prev, rewards_prev, training=None, reset_state=None
    ):
        hidden = self.get_embedding(
            observations,
            actions_prev,
            rewards_prev,
            training=training,
            reset_state=reset_state,
        )

        action_logits = self.dense_action_logits(hidden)
        direction_logits = self.dense_direction_logits(hidden)

        action_dist = tfp.distributions.Categorical(logits=action_logits)
        direction_dist = tfp.distributions.Categorical(logits=direction_logits)
        dist = pynr.distributions.MultiCategorical([action_dist, direction_dist])

        return dist
