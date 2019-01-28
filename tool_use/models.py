import pyoneer as pynr
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import tensorflow_probability as tfp


class PolicyValue(tf.keras.Model):
    def __init__(self, action_size, scale, **kwargs):
        super(PolicyValue, self).__init__(**kwargs)

        kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2.0)
        logits_initializer = tf.keras.initializers.VarianceScaling(scale=1.0)
        scale_initializer = pynr.initializers.SoftplusInverse(scale=scale)

        self.dense_hidden1 = tf.keras.layers.Dense(
            units=64,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)
        self.dense_hidden2 = tf.keras.layers.Dense(
            units=64,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)
        self.dense_loc_hidden = tf.keras.layers.Dense(
            units=64,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)
        self.dense_loc = tf.keras.layers.Dense(
            units=action_size,
            activation=tf.math.tanh,
            kernel_initializer=logits_initializer)
        self.dense_value_hidden = tf.keras.layers.Dense(
            units=64,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)
        self.dense_value = tf.keras.layers.Dense(
            units=1, activation=None, kernel_initializer=logits_initializer)

        self.scale_diag_inverse = tfe.Variable(
            scale_initializer(action_size), trainable=True)

    @property
    def scale_diag(self):
        return tf.nn.softplus(self.scale_diag_inverse)

    def _forward(self, inputs, training=None, reset_state=None):
        inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
        hidden = self.dense_hidden1(inputs)
        hidden = self.dense_hidden2(hidden)
        return hidden

    @tfe.defun
    def get_values(self, inputs, training=None, reset_state=None):
        hidden = self._forward(
            inputs, training=training, reset_state=reset_state)
        hidden = self.dense_value_hidden(hidden)
        value = self.dense_value(hidden)
        return value[..., 0]

    @tfe.defun
    def call(self, inputs, training=None, reset_state=None):
        hidden = self._forward(
            inputs, training=training, reset_state=reset_state)
        hidden = self.dense_loc_hidden(hidden)
        loc = self.dense_loc(hidden)
        dist = tfp.distributions.MultivariateNormalDiag(
            loc=loc, scale_diag=self.scale_diag)
        return dist
