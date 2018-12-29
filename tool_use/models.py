import pyoneer as pynr
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import tensorflow_probability as tfp


class Policy(tf.keras.Model):
    def __init__(self, observation_space, action_space, scale, **kwargs):
        super(Policy, self).__init__(**kwargs)

        self.observation_space = observation_space
        self.action_space = action_space

        kernel_initializer = tf.initializers.variance_scaling(scale=2.0)
        logits_initializer = tf.initializers.variance_scaling(scale=1.0)

        self.dense1 = tf.keras.layers.Dense(
            units=64,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)
        self.dense2 = tf.keras.layers.Dense(
            units=64,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)

        action_size = action_space.shape[0]
        self.dense_loc = tf.keras.layers.Dense(
            units=action_size,
            activation=tf.tanh,
            kernel_initializer=logits_initializer)
        self.scale_diag_inverse = tfe.Variable(
            tfp.distributions.softplus_inverse([scale] * action_size),
            trainable=True)

    @property
    def scale_diag(self):
        return tf.nn.softplus(self.scale_diag_inverse)

    def call(self, inputs, training=None, reset_state=None):
        high = self.observation_space.high
        low = self.observation_space.low

        high = tf.where(tf.is_finite(high), high, tf.ones_like(high))
        low = tf.where(tf.is_finite(low), low, -tf.ones_like(low))

        inputs = pynr.math.high_low_normalize(inputs, low=low, high=high)

        hidden = self.dense1(inputs)
        hidden = self.dense2(hidden)

        loc = self.dense_loc(hidden)
        loc = pynr.math.rescale(
            loc,
            oldmin=-1,
            oldmax=1,
            newmin=self.action_space.low,
            newmax=self.action_space.high)

        return tfp.distributions.MultivariateNormalDiag(
            loc=loc, scale_diag=self.scale_diag)


class Value(tf.keras.Model):
    def __init__(self, observation_space, action_space, **kwargs):
        super(Value, self).__init__(**kwargs)

        self.observation_space = observation_space
        self.action_space = action_space

        kernel_initializer = tf.initializers.variance_scaling(scale=2.0)
        logits_initializer = tf.initializers.variance_scaling(scale=1.0)

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

    def call(self, states, actions, training=None, reset_state=None):
        high = tf.where(
            tf.is_finite(self.observation_space.high),
            self.observation_space.high,
            tf.ones_like(self.observation_space.high))
        low = tf.where(
            tf.is_finite(self.observation_space.low),
            self.observation_space.low,
            -tf.ones_like(self.observation_space.low))
        states_norm = pynr.math.high_low_normalize(states, low=low, high=high)

        # high = tf.where(
        #     tf.is_finite(self.action_space.high), self.action_space.high,
        #     tf.ones_like(self.action_space.high))
        # low = tf.where(
        #     tf.is_finite(self.action_space.low), self.action_space.low,
        #     -tf.ones_like(self.action_space.low))
        # actions_norm = pynr.math.high_low_normalize(
        #     actions, low=low, high=high)

        inputs = tf.concat([states_norm], axis=-1)

        hidden = self.dense1(inputs)
        hidden = self.dense2(hidden)

        value = self.dense_value(hidden)
        value = tf.squeeze(value, axis=-1)
        return value
