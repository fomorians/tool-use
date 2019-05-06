import pyoneer as pynr
import tensorflow as tf
import tensorflow_probability as tfp


class PolicyModel(tf.keras.Model):

    embedding_size = 128

    def __init__(self, action_size):
        super(PolicyModel, self).__init__()

        kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2.0)
        logits_initializer = tf.keras.initializers.VarianceScaling(scale=1.0)

        self.dense_hidden = tf.keras.layers.Dense(
            units=self.embedding_size,
            activation=pynr.nn.swish,
            use_bias=True,
            kernel_initializer=kernel_initializer,
        )
        self.dense_logits = tf.keras.layers.Dense(
            units=action_size, activation=None, kernel_initializer=logits_initializer
        )

    @property
    def scale_diag(self):
        return tf.nn.softplus(self.scale_diag_inverse)

    def call(self, observations, training=None):
        observations_flat = tf.reshape(
            observations,
            [
                observations.shape[0],
                observations.shape[1],
                observations.shape[2] * observations.shape[3] * observations.shape[4],
            ],
        )
        hidden = self.dense_hidden(observations_flat)
        logits = self.dense_logits(hidden)
        dist = tfp.distributions.Categorical(logits=logits)
        return dist


class ValueModel(tf.keras.Model):

    embedding_size = 128

    def __init__(self):
        super(ValueModel, self).__init__()

        kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2.0)
        logits_initializer = tf.keras.initializers.VarianceScaling(scale=1.0)

        self.dense_hidden = tf.keras.layers.Dense(
            units=self.embedding_size,
            activation=pynr.nn.swish,
            use_bias=True,
            kernel_initializer=kernel_initializer,
        )
        self.dense_values = tf.keras.layers.Dense(
            units=1, activation=None, kernel_initializer=logits_initializer
        )

    def call(self, observations, training=None):
        observations_flat = tf.reshape(
            observations,
            [
                observations.shape[0],
                observations.shape[1],
                observations.shape[2] * observations.shape[3] * observations.shape[4],
            ],
        )
        hidden = self.dense_hidden(observations_flat)
        values = self.dense_values(hidden)
        return values[..., 0]
