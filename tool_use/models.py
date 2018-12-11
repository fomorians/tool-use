import tensorflow as tf
import tensorflow_probability as tfp


class Policy(tf.keras.Model):
    def __init__(self, action_size):
        super(Policy, self).__init__()
        self.linear = tf.layers.Dense(units=action_size)

    def call(self, inputs, training=None):
        return tfp.distributions.MultivariateNormalDiag(self.linear(inputs))


class Value(tf.keras.Model):
    def __init__(self):
        super(Value, self).__init__()
        self.linear = tf.layers.Dense(units=1)

    def call(self, inputs, training=None):
        return self.linear(inputs)
