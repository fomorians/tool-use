import pyoneer as pynr
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import tensorflow_probability as tfp


class Policy(tf.keras.Model):
    def __init__(self, action_size, scale, **kwargs):
        super(Policy, self).__init__(**kwargs)

        kernel_initializer = tf.initializers.variance_scaling(scale=2.0)

        self.dense1 = tf.keras.layers.Dense(
            units=200, kernel_initializer=kernel_initializer)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(
            units=100, kernel_initializer=kernel_initializer)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.dense_logits = tf.keras.layers.Dense(
            units=action_size, kernel_initializer=kernel_initializer)
        self.scale_diag_inverse = tfe.Variable(
            tfp.distributions.softplus_inverse([scale] * action_size),
            trainable=True)

    @property
    def scale_diag(self):
        return tf.nn.softplus(self.scale_diag_inverse)

    def call(self, inputs, training=None):
        hidden1 = pynr.nn.swish(self.bn1(self.dense1(inputs), training=False))
        hidden2 = pynr.nn.swish(self.bn2(self.dense2(hidden1), training=False))
        loc = self.dense_logits(hidden2)
        return tfp.distributions.MultivariateNormalDiag(
            loc=loc, scale_diag=self.scale_diag)


class Value(tf.keras.Model):
    def __init__(self):
        super(Value, self).__init__()

        kernel_initializer = tf.initializers.variance_scaling(scale=2.0)

        self.dense1 = tf.keras.layers.Dense(
            units=200, kernel_initializer=kernel_initializer)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(
            units=100, kernel_initializer=kernel_initializer)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.dense_logits = tf.keras.layers.Dense(
            units=1, kernel_initializer=kernel_initializer)

    def call(self, inputs, training=None):
        hidden1 = pynr.nn.swish(self.bn1(self.dense1(inputs), training=False))
        hidden2 = pynr.nn.swish(self.bn2(self.dense2(hidden1), training=False))
        value = self.dense_logits(hidden2)
        return value[..., 0]
