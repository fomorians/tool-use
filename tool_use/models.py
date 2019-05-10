import pyoneer as pynr
import tensorflow as tf
import tensorflow_probability as tfp


class MultiCategorical:
    def __init__(self, distributions):
        self.distributions = distributions

    def log_prob(self, value):
        values = tf.split(value, len(self.distributions), axis=-1)
        return tf.math.add_n(
            [
                dist.log_prob(val[..., 0])
                for dist, val in zip(self.distributions, values)
            ]
        )

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
        batch_size, steps = observations.shape[:2]
        observations_flat = tf.reshape(observations, [batch_size, steps, 12 * 12 * 3])

        hidden = self.dense_hidden(observations_flat)

        action_logits = self.dense_action_logits(hidden)
        direction_logits = self.dense_direction_logits(hidden)

        action_dist = tfp.distributions.Categorical(logits=action_logits)
        direction_dist = tfp.distributions.Categorical(logits=direction_logits)
        dist = MultiCategorical([action_dist, direction_dist])
        return dist


class ValueModel(tf.keras.Model):
    def __init__(self):
        super(ValueModel, self).__init__()

        kernel_initializer = tf.initializers.VarianceScaling(scale=2.0)
        logits_initializer = tf.initializers.VarianceScaling(scale=1.0)

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
        batch_size, steps = observations.shape[:2]
        observations_flat = tf.reshape(observations, [batch_size, steps, 12 * 12 * 3])

        hidden = self.dense_hidden(observations_flat)
        values = self.dense_logits(hidden)
        return values[..., 0]
