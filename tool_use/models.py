import pyoneer as pynr
import tensorflow as tf
import tensorflow_probability as tfp


# TODO: move to pyoneer
class MultiCategorical:
    def __init__(self, distributions):
        self.distributions = distributions

    def log_prob(self, value):
        values = tf.split(value, len(self.distributions), axis=-1)
        log_probs = []
        for dist, val in zip(self.distributions, values):
            log_prob = dist.log_prob(val[..., 0])
            log_probs.append(log_prob)
        return tf.math.add_n(log_probs)

    def entropy(self):
        return tf.math.add_n([dist.entropy() for dist in self.distributions])

    def sample(self):
        return tf.stack([dist.sample() for dist in self.distributions], axis=-1)

    def mode(self):
        return tf.stack([dist.mode() for dist in self.distributions], axis=-1)


class Model(tf.keras.Model):
    def __init__(self, action_space):
        super(Model, self).__init__()

        kernel_initializer = tf.initializers.VarianceScaling(scale=2.0)
        logits_initializer = tf.initializers.VarianceScaling(scale=1.0)

        self.initial_hidden_state = tf.Variable(tf.zeros(shape=[64]), trainable=True)
        self.initial_cell_state = tf.Variable(tf.zeros(shape=[64]), trainable=True)

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

    def get_embedding(self, observations, training=None, reset_state=None):
        batch_size, steps, height, width, channels = observations.shape

        # TODO: implement contextual reshapes
        # with FlattenDims(axis=[0, 1]):
        #     pass
        # with ExpandDims(axis=[0]):
        #     pass
        observations = tf.reshape(
            observations, [batch_size * steps, height, width, channels]
        )

        hidden = self.conv2d_hidden1(observations)
        hidden = self.conv2d_hidden2(hidden)
        hidden = self.flatten(hidden)
        hidden = self.dense_hidden(hidden)

        hidden = tf.reshape(hidden, [batch_size, steps, self.rnn.units])

        if self.hidden_state is None or self.cell_state is None or reset_state:
            self.hidden_state = tf.tile(
                self.initial_hidden_state[None, ...], [batch_size, 1]
            )
            self.cell_state = tf.tile(
                self.initial_cell_state[None, ...], [batch_size, 1]
            )

        hidden, self.hidden_state, self.cell_state = self.rnn(
            hidden, initial_state=(self.hidden_state, self.cell_state)
        )
        return hidden

    def get_distribution(self, observations, training=None, reset_state=None):
        hidden = self.get_embedding(
            observations, training=training, reset_state=reset_state
        )

        action_logits = self.dense_action_logits(hidden)
        direction_logits = self.dense_direction_logits(hidden)

        action_dist = tfp.distributions.Categorical(logits=action_logits)
        direction_dist = tfp.distributions.Categorical(logits=direction_logits)
        dist = MultiCategorical([action_dist, direction_dist])

        return dist

    def call(self, observations, training=None, reset_state=None):
        hidden = self.get_embedding(
            observations, training=training, reset_state=reset_state
        )

        action_logits = self.dense_action_logits(hidden)
        direction_logits = self.dense_direction_logits(hidden)
        values_logits = self.dense_values_logits(hidden)

        action_dist = tfp.distributions.Categorical(logits=action_logits)
        direction_dist = tfp.distributions.Categorical(logits=direction_logits)
        dist = MultiCategorical([action_dist, direction_dist])

        return dist, values_logits[..., 0]
