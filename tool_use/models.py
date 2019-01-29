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
        kernel_regularizer = tf.keras.regularizers.l2(1e-4)

        # self.initial_hidden_state = tfe.Variable(
        #     tf.zeros(shape=[64]), trainable=True)
        # self.hidden_state = None

        self.dense_hidden1 = tf.keras.layers.Dense(
            units=64,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)
        self.dense_hidden2 = tf.keras.layers.Dense(
            units=64,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)
        # self.rnn_hidden = tf.keras.layers.GRU(
        #     units=64, return_sequences=True, return_state=True)
        self.dense_hidden_loc = tf.keras.layers.Dense(
            units=64,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)
        self.dense_loc = tf.keras.layers.Dense(
            units=action_size,
            activation=tf.math.tanh,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)
        self.dense_hidden_values = tf.keras.layers.Dense(
            units=64,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)
        self.dense_values = tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=logits_initializer,
            kernel_regularizer=kernel_regularizer)

        self.scale_diag_inverse = tfe.Variable(
            scale_initializer(action_size), trainable=True)

    @property
    def scale_diag(self):
        return tf.nn.softplus(self.scale_diag_inverse)

    def _get_embedding(self, observations, training=None, reset_state=None):
        observations = tf.convert_to_tensor(observations, dtype=self.dtype)

        # TODO: separate ego-centric features
        hidden = self.dense_hidden1(observations)
        hidden = self.dense_hidden2(hidden)

        # if self.hidden_state is None or reset_state:
        #     batch_size = observations.shape[0]
        #     self.hidden_state = tf.tile(self.initial_hidden_state[None, ...],
        #                                 [batch_size, 1])

        # hidden, self.hidden_state = self.rnn_hidden(
        #     hidden, initial_state=[self.hidden_state])

        return hidden

    def _get_dist(self, embedding, training=None):
        hidden = self.dense_hidden_loc(embedding)
        loc = self.dense_loc(hidden)
        dist = tfp.distributions.MultivariateNormalDiag(
            loc=loc, scale_diag=self.scale_diag)
        return dist

    def _get_values(self, embedding, training=None):
        hidden = self.dense_hidden_values(embedding)
        values = self.dense_values(hidden)
        return values[..., 0]

    @tfe.defun
    def forward(self,
                observations,
                actions=None,
                training=None,
                reset_state=None,
                include=None):
        if include is None:
            include = ['log_probs', 'entropy', 'values']

        embedding = self._get_embedding(
            observations, training=training, reset_state=reset_state)

        predictions = {}

        if 'values' in include:
            predictions['values'] = self._get_values(
                embedding, training=training)

        if 'log_probs' in include or 'entropy' in include:
            dist = self._get_dist(embedding, training=training)

            if 'log_probs' in include:
                assert actions is not None
                predictions['log_probs'] = dist.log_prob(actions)

            if 'entropy' in include:
                predictions['entropy'] = dist.entropy()

        return predictions

    def call(self, observations, training=None, reset_state=None):
        embedding = self._get_embedding(
            observations, training=training, reset_state=reset_state)
        dist = self._get_dist(embedding, training=training)
        return dist
