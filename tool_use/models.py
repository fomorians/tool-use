import pyoneer as pynr
import tensorflow as tf
import tensorflow_probability as tfp


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


class PolicyModel(tf.keras.Model):
    def __init__(self, action_space):
        super(PolicyModel, self).__init__()

        kernel_initializer = tf.initializers.VarianceScaling(scale=2.0)
        logits_initializer = tf.initializers.VarianceScaling(scale=1.0)

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
            units=128,
            activation=pynr.nn.swish,
            use_bias=True,
            kernel_initializer=kernel_initializer,
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

    def call(self, observations, training=None):
        batch_size, steps, height, width, channels = observations.shape
        observations = tf.reshape(
            observations, [batch_size * steps, height, width, channels]
        )

        hidden = self.conv2d_hidden1(observations)
        hidden = self.conv2d_hidden2(hidden)
        hidden = self.flatten(hidden)
        hidden = self.dense_hidden(hidden)

        action_logits = self.dense_action_logits(hidden)
        direction_logits = self.dense_direction_logits(hidden)

        action_logits = tf.reshape(action_logits, [batch_size, steps, -1])
        direction_logits = tf.reshape(direction_logits, [batch_size, steps, -1])

        action_dist = tfp.distributions.Categorical(logits=action_logits)
        direction_dist = tfp.distributions.Categorical(logits=direction_logits)
        dist = MultiCategorical([action_dist, direction_dist])
        return dist


class ValueModel(tf.keras.Model):
    def __init__(self):
        super(ValueModel, self).__init__()

        kernel_initializer = tf.initializers.VarianceScaling(scale=2.0)
        logits_initializer = tf.initializers.VarianceScaling(scale=1.0)

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
            units=128,
            activation=pynr.nn.swish,
            use_bias=True,
            kernel_initializer=kernel_initializer,
        )
        self.dense_logits = tf.keras.layers.Dense(
            units=1, activation=None, kernel_initializer=logits_initializer
        )

    def call(self, observations, training=None):
        batch_size, steps, height, width, channels = observations.shape
        observations = tf.reshape(
            observations, [batch_size * steps, height, width, channels]
        )

        hidden = self.conv2d_hidden1(observations)
        hidden = self.conv2d_hidden2(hidden)
        hidden = self.flatten(hidden)
        hidden = self.dense_hidden(hidden)

        values = self.dense_logits(hidden)

        values = tf.reshape(values, [batch_size, steps, -1])

        return values[..., 0]
