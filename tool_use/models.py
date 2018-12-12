import pyoneer as pynr
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import tensorflow_probability as tfp


class Policy(tf.keras.Model):
    def __init__(self, action_size, **kwargs):
        super(Policy, self).__init__(**kwargs)

        kernel_initializer = tf.initializers.variance_scaling(scale=2.0)

        self.dense1 = tf.layers.Dense(
            units=64,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)
        self.dense1_static = tf.layers.Dense(
            units=64,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer,
            trainable=False)
        self.dense2 = tf.layers.Dense(
            units=32,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)
        self.dense2_static = tf.layers.Dense(
            units=32,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer,
            trainable=False)
        self.dense3 = tf.layers.Dense(
            units=action_size,
            activation=tf.tanh,
            kernel_initializer=kernel_initializer)
        self.dense3_static = tf.layers.Dense(
            units=action_size,
            activation=tf.tanh,
            kernel_initializer=kernel_initializer,
            trainable=False)
        self.scale_diag_inverse = tfe.Variable(
            tfp.distributions.softplus_inverse([1.0] * action_size),
            trainable=True)

    def call(self, inputs, training=None):
        hidden = self.dense1(inputs) + self.dense1_static(inputs)
        hidden = self.dense2(hidden) + self.dense2_static(hidden)
        loc = self.dense3(hidden) + self.dense3_static(hidden)
        scale_diag = tf.nn.softplus(self.scale_diag_inverse)
        return tfp.distributions.MultivariateNormalDiag(
            loc=loc, scale_diag=scale_diag)


class Value(tf.keras.Model):
    def __init__(self):
        super(Value, self).__init__()

        kernel_initializer = tf.initializers.variance_scaling(scale=2.0)

        self.dense1 = tf.layers.Dense(
            units=64,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)
        self.dense2 = tf.layers.Dense(
            units=32,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)
        self.dense3 = tf.layers.Dense(
            units=1, kernel_initializer=kernel_initializer)

    def call(self, inputs, training=None):
        hidden = self.dense1(inputs)
        hidden = self.dense2(hidden)
        value = self.dense3(hidden)
        return value
