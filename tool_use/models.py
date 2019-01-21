import pyoneer as pynr
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import tensorflow_probability as tfp


class SquashedMultivariateNormalDiag(tfp.distributions.MultivariateNormalDiag):
    def mode(self):
        mode = super(SquashedMultivariateNormalDiag, self).mode()
        return tf.tanh(mode)

    def sample(self):
        sample = super(SquashedMultivariateNormalDiag, self).sample()
        return tf.tanh(sample)

    def log_prob(self, value):
        raw_value = tf.atanh(value)
        log_probs = super(SquashedMultivariateNormalDiag,
                          self).log_prob(raw_value)
        log_probs -= tf.reduce_sum(tf.log(1 - value**2 + 1e-6), axis=-1) + 1e-6
        return log_probs


class Policy(tf.keras.Model):
    def __init__(self, observation_space, action_space, scale, **kwargs):
        super(Policy, self).__init__(**kwargs)

        self.observation_space = observation_space
        self.action_space = action_space

        # self.initial_hidden_state = tfe.Variable(
        #     tf.zeros(shape=[64]), trainable=True)
        # self.initial_cell_state = tfe.Variable(
        #     tf.zeros(shape=[64]), trainable=True)

        kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2.0)
        logits_initializer = tf.keras.initializers.VarianceScaling(scale=1.0)
        scale_initializer = pynr.initializers.SoftplusInverse(scale=scale)

        self.dense1 = tf.keras.layers.Dense(
            units=64,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)
        # self.rnn = tf.keras.layers.LSTM(
        #     units=64, return_sequences=True, return_state=True)
        self.dense2 = tf.keras.layers.Dense(
            units=64,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)

        self.dense_loc = tf.keras.layers.Dense(
            units=self.action_space.shape[0],
            activation=None,
            kernel_initializer=logits_initializer)
        self.scale_diag_inverse = tfe.Variable(
            scale_initializer(self.action_space.shape), trainable=True)

        # self.hidden_state = None
        # self.cell_state = None

    @property
    def scale_diag(self):
        return tf.nn.softplus(self.scale_diag_inverse)

    # def get_initial_state(self, batch_size):
    #     """
    #     Return a properly shaped initial state for the RNN.
    #     Args:
    #         batch_size: The size of the first dimension of the inputs
    #                     (excluding time steps).
    #     """
    #     hidden_state = tf.tile(self.initial_hidden_state[None, ...],
    #                            [batch_size, 1])
    #     cell_state = tf.tile(self.initial_cell_state[None, ...],
    #                          [batch_size, 1])
    #     return hidden_state, cell_state

    def call(self, inputs, training=None, reset_state=None):
        loc, var = pynr.nn.moments_from_range(
            minval=self.observation_space.low,
            maxval=self.observation_space.high)
        inputs = pynr.math.normalize(inputs, loc=loc, scale=tf.sqrt(var))

        hidden = self.dense1(inputs)

        # # compute an initial state for the RNN
        # if self.hidden_state is None or self.cell_state is None or reset_state:
        #     self.hidden_state, self.cell_state = self.get_initial_state(
        #         inputs.shape[0])

        # hidden, self.hidden_state, self.cell_state = self.rnn(
        #     hidden, initial_state=[self.hidden_state, self.cell_state])

        hidden = self.dense2(hidden)
        loc = self.dense_loc(hidden)

        dist = tfp.distributions.MultivariateNormalDiag(
            loc=loc, scale_diag=self.scale_diag)
        return dist


class Value(tf.keras.Model):
    def __init__(self, observation_space, **kwargs):
        super(Value, self).__init__(**kwargs)

        self.observation_space = observation_space

        # self.initial_hidden_state = tfe.Variable(
        #     tf.zeros(shape=[64]), trainable=True)
        # self.initial_cell_state = tfe.Variable(
        #     tf.zeros(shape=[64]), trainable=True)

        kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2.0)
        logits_initializer = tf.keras.initializers.VarianceScaling(scale=1.0)

        self.dense1 = tf.keras.layers.Dense(
            units=64,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)
        # self.rnn = tf.keras.layers.LSTM(
        #     units=64, return_sequences=True, return_state=True)
        self.dense2 = tf.keras.layers.Dense(
            units=64,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)
        self.dense_value = tf.keras.layers.Dense(
            units=1, activation=None, kernel_initializer=logits_initializer)

        # self.hidden_state = None
        # self.cell_state = None

    # def get_initial_state(self, batch_size):
    #     """
    #     Return a properly shaped initial state for the RNN.
    #     Args:
    #         batch_size: The size of the first dimension of the inputs
    #                     (excluding time steps).
    #     """
    #     hidden_state = tf.tile(self.initial_hidden_state[None, ...],
    #                            [batch_size, 1])
    #     cell_state = tf.tile(self.initial_cell_state[None, ...],
    #                          [batch_size, 1])
    #     return hidden_state, cell_state

    def call(self, inputs, training=None, reset_state=None):
        loc, var = pynr.nn.moments_from_range(
            minval=self.observation_space.low,
            maxval=self.observation_space.high)
        inputs = pynr.math.normalize(inputs, loc=loc, scale=tf.sqrt(var))

        hidden = self.dense1(inputs)

        # # compute an initial state for the RNN
        # if self.hidden_state is None or self.cell_state is None or reset_state:
        #     self.hidden_state, self.cell_state = self.get_initial_state(
        #         inputs.shape[0])

        # hidden, self.hidden_state, self.cell_state = self.rnn(
        #     hidden, initial_state=[self.hidden_state, self.cell_state])

        hidden = self.dense2(hidden)
        value = self.dense_value(hidden)

        return value[..., 0]
