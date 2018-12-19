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
        logits_initializer = tf.initializers.variance_scaling(scale=2.0)

        self.initial_hidden_state = tfe.Variable(
            tf.zeros(shape=[64]), trainable=True)
        self.initial_cell_state = tfe.Variable(
            tf.zeros(shape=[64]), trainable=True)

        self.dense1 = tf.keras.layers.Dense(
            units=64,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)
        self.dense2 = tf.keras.layers.Dense(
            units=64,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)

        self.rnn = tf.keras.layers.LSTM(
            units=64, return_sequences=True, return_state=True)

        action_size = action_space.shape[0]
        self.dense_loc = tf.keras.layers.Dense(
            units=action_size,
            activation=None,
            kernel_initializer=logits_initializer)

        self.scale_diag_inverse = tfe.Variable(
            tfp.distributions.softplus_inverse([scale] * action_size),
            trainable=True)

        self.hidden_state = None
        self.cell_state = None

    @property
    def scale_diag(self):
        return tf.nn.softplus(self.scale_diag_inverse)

    def get_initial_state(self, batch_size):
        hidden_state = tf.tile(self.initial_hidden_state[None, ...],
                               [batch_size, 1])
        cell_state = tf.tile(self.initial_cell_state[None, ...],
                             [batch_size, 1])
        return hidden_state, cell_state

    def call(self, inputs, training=None, reset_state=None):
        hidden = self.dense1(inputs)
        hidden = self.dense2(hidden)

        if self.hidden_state is None or self.cell_state is None or reset_state:
            self.hidden_state, self.cell_state = self.get_initial_state(
                inputs.shape[0])

        hidden, self.hidden_state, self.cell_state = self.rnn(
            hidden, initial_state=[self.hidden_state, self.cell_state])

        loc = self.dense_loc(hidden)
        return tfp.distributions.MultivariateNormalDiag(
            loc=loc, scale_diag=self.scale_diag)


class Value(tf.keras.Model):
    def __init__(self, observation_space, **kwargs):
        super(Value, self).__init__(**kwargs)

        self.observation_space = observation_space

        kernel_initializer = tf.initializers.variance_scaling(scale=2.0)
        logits_initializer = tf.initializers.variance_scaling(scale=2.0)

        self.initial_hidden_state = tfe.Variable(
            tf.zeros(shape=[64]), trainable=True)
        self.initial_cell_state = tfe.Variable(
            tf.zeros(shape=[64]), trainable=True)

        self.dense1 = tf.keras.layers.Dense(
            units=64,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)
        self.dense2 = tf.keras.layers.Dense(
            units=64,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)

        self.rnn = tf.keras.layers.LSTM(
            units=64, return_sequences=True, return_state=True)

        self.dense_value = tf.keras.layers.Dense(
            units=1, activation=None, kernel_initializer=logits_initializer)

        self.hidden_state = None
        self.cell_state = None

    def get_initial_state(self, batch_size):
        hidden_state = tf.tile(self.initial_hidden_state[None, ...],
                               [batch_size, 1])
        cell_state = tf.tile(self.initial_cell_state[None, ...],
                             [batch_size, 1])
        return hidden_state, cell_state

    def call(self, inputs, training=None, reset_state=None):
        hidden = self.dense1(inputs)
        hidden = self.dense2(hidden)

        if self.hidden_state is None or self.cell_state is None or reset_state:
            self.hidden_state, self.cell_state = self.get_initial_state(
                inputs.shape[0])

        hidden, self.hidden_state, self.cell_state = self.rnn(
            hidden, initial_state=[self.hidden_state, self.cell_state])

        value = self.dense_value(hidden)
        return tf.squeeze(value, axis=-1)
