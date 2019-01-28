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
        self.dense_hidden_loc = tf.keras.layers.Dense(
            units=64,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)
        self.dense_loc = tf.keras.layers.Dense(
            units=action_size,
            activation=tf.math.tanh,
            kernel_initializer=logits_initializer)
        self.dense_hidden_values = tf.keras.layers.Dense(
            units=64,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)
        self.dense_values = tf.keras.layers.Dense(
            units=1, activation=None, kernel_initializer=logits_initializer)

        self.scale_diag_inverse = tfe.Variable(
            scale_initializer(action_size), trainable=True)

    @property
    def scale_diag(self):
        return tf.nn.softplus(self.scale_diag_inverse)

    @tfe.defun
    def forward(self,
                observations,
                actions=None,
                training=None,
                reset_state=None,
                include=None):
        if include is None:
            include = ['log_probs', 'entropy', 'values']

        observations = tf.convert_to_tensor(observations, dtype=self.dtype)

        hidden_shared = self.dense_hidden1(observations)
        hidden_shared = self.dense_hidden2(hidden_shared)

        predictions = {}

        if 'values' in include:
            hidden_values = self.dense_hidden_values(hidden_shared)
            values = self.dense_values(hidden_values)
            predictions['values'] = values[..., 0]

        if 'log_probs' in include or 'entropy' in include:
            hidden_loc = self.dense_hidden_loc(hidden_shared)
            loc = self.dense_loc(hidden_loc)
            dist = tfp.distributions.MultivariateNormalDiag(
                loc=loc, scale_diag=self.scale_diag)

            if 'log_probs' in include:
                assert actions is not None
                predictions['log_probs'] = dist.log_prob(actions)

            if 'entropy' in include:
                predictions['entropy'] = dist.entropy()

        return predictions

    def call(self, observations, training=None, reset_state=None):
        observations = tf.convert_to_tensor(observations, dtype=self.dtype)
        hidden_shared = self.dense_hidden1(observations)
        hidden_shared = self.dense_hidden2(hidden_shared)
        hidden_loc = self.dense_hidden_loc(hidden_shared)
        loc = self.dense_loc(hidden_loc)
        dist = tfp.distributions.MultivariateNormalDiag(
            loc=loc, scale_diag=self.scale_diag)
        return dist
