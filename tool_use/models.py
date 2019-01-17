import pyoneer as pynr
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import tensorflow_probability as tfp


class Policy(tf.keras.Model):
    def __init__(self, observation_space, action_space, scale, **kwargs):
        super(Policy, self).__init__(**kwargs)

        self.observation_space = observation_space
        self.action_space = action_space

        kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2.0)
        logits_initializer = tf.keras.initializers.VarianceScaling(scale=1.0)
        scale_initializer = pynr.initializers.SoftplusInverse(scale=scale)

        self.dense1 = tf.keras.layers.Dense(
            units=64,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)
        self.dense2 = tf.keras.layers.Dense(
            units=64,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)

        self.dense_loc = tf.keras.layers.Dense(
            units=self.action_space.shape[0],
            activation=tf.tanh,
            kernel_initializer=logits_initializer)
        self.scale_diag_inverse = tfe.Variable(
            scale_initializer(self.action_space.shape), trainable=True)

    @property
    def scale_diag(self):
        return tf.nn.softplus(self.scale_diag_inverse)

    def call(self, inputs, training=None, reset_state=None):
        high = self.observation_space.high
        low = self.observation_space.low

        high = tf.where(tf.is_finite(high), high, tf.ones_like(high))
        low = tf.where(tf.is_finite(low), low, -tf.ones_like(low))

        loc, var = pynr.nn.moments_from_range(low, high)
        inputs = pynr.math.normalize(inputs, loc=loc, scale=tf.sqrt(var))

        hidden = self.dense1(inputs)
        hidden = self.dense2(hidden)

        loc = pynr.math.rescale(
            self.dense_loc(hidden),
            oldmin=-1.0,
            oldmax=1.0,
            newmin=self.action_space.low,
            newmax=self.action_space.high)

        dist = tfp.distributions.MultivariateNormalDiag(
            loc=loc, scale_diag=self.scale_diag)
        return dist


class Value(tf.keras.Model):
    def __init__(self, observation_space, **kwargs):
        super(Value, self).__init__(**kwargs)

        self.observation_space = observation_space

        kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2.0)
        logits_initializer = tf.keras.initializers.VarianceScaling(scale=1.0)

        self.dense1 = tf.keras.layers.Dense(
            units=64,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)
        self.dense2 = tf.keras.layers.Dense(
            units=64,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)
        self.dense_value = tf.keras.layers.Dense(
            units=1, activation=None, kernel_initializer=logits_initializer)

    def call(self, inputs, training=None, reset_state=None):
        high = self.observation_space.high
        low = self.observation_space.low

        high = tf.where(tf.is_finite(high), high, tf.ones_like(high))
        low = tf.where(tf.is_finite(low), low, -tf.ones_like(low))

        loc, var = pynr.nn.moments_from_range(low, high)
        inputs = pynr.math.normalize(inputs, loc=loc, scale=tf.sqrt(var))

        hidden = self.dense1(inputs)
        hidden = self.dense2(hidden)
        value = self.dense_value(hidden)

        return value[..., 0]
