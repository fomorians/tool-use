import pyoneer as pynr
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import tensorflow_probability as tfp


class StateModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(StateModel, self).__init__(**kwargs)

        kernel_initializer = tf.initializers.variance_scaling(scale=2.0)
        logits_initializer = tf.initializers.variance_scaling(scale=1.0)

        self.bn = tf.keras.layers.BatchNormalization()
        self.dense1 = tf.keras.layers.Dense(
            units=128,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)
        self.dense2 = tf.keras.layers.Dense(
            units=64,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)
        self.dense_logits = tf.keras.layers.Dense(
            units=32, activation=None, kernel_initializer=logits_initializer)

    def call(self, inputs, training=None):
        inputs = self.bn(inputs)
        hidden = self.dense1(inputs)
        hidden = self.dense2(hidden)
        value = self.dense_logits(hidden)
        return value


class Policy(tf.keras.Model):
    def __init__(self, action_size, **kwargs):
        super(Policy, self).__init__(**kwargs)

        kernel_initializer = tf.initializers.variance_scaling(scale=2.0)

        self.bn = tf.keras.layers.BatchNormalization()
        self.dense1 = tf.keras.layers.Dense(
            units=128,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)
        self.dense2 = tf.keras.layers.Dense(
            units=64,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)
        self.dense_logits = tf.keras.layers.Dense(
            units=action_size,
            activation=tf.tanh,
            kernel_initializer=kernel_initializer)
        self.scale_diag_inverse = tfe.Variable(
            tfp.distributions.softplus_inverse([1.0] * action_size),
            trainable=True)

    def call(self, inputs, training=None):
        inputs = self.bn(inputs)
        hidden = self.dense1(inputs)
        hidden = self.dense2(hidden)
        loc = self.dense_logits(hidden)
        scale_diag = tf.nn.softplus(self.scale_diag_inverse)
        return tfp.distributions.MultivariateNormalDiag(
            loc=loc, scale_diag=scale_diag)


class Value(tf.keras.Model):
    def __init__(self):
        super(Value, self).__init__()

        kernel_initializer = tf.initializers.variance_scaling(scale=2.0)

        self.bn = tf.keras.layers.BatchNormalization()
        self.dense1 = tf.keras.layers.Dense(
            units=128,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)
        self.dense2 = tf.keras.layers.Dense(
            units=64,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)
        self.dense_logits = tf.keras.layers.Dense(
            units=1, kernel_initializer=kernel_initializer)

    def call(self, inputs, training=None):
        inputs = self.bn(inputs)
        hidden = self.dense1(inputs)
        hidden = self.dense2(hidden)
        value = self.dense_logits(hidden)
        return value
