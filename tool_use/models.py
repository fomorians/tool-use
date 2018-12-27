import pyoneer as pynr
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import tensorflow_probability as tfp


class StateEmbedding(tf.keras.Model):
    def __init__(self, observation_space, **kwargs):
        super(StateEmbedding, self).__init__(**kwargs)

        self.observation_space = observation_space

        kernel_initializer = tf.initializers.variance_scaling(scale=2.0)

        self.dense_hidden = tf.keras.layers.Dense(
            units=64,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)
        self.dense_embedding = tf.keras.layers.Dense(
            units=64,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)

    def call(self, inputs, training=None, reset_state=None):
        high = self.observation_space.high
        low = self.observation_space.low

        high = tf.where(tf.is_finite(high), high, tf.ones_like(high))
        low = tf.where(tf.is_finite(low), low, -tf.ones_like(low))

        inputs = pynr.math.high_low_normalize(inputs, low=low, high=high)

        hidden = self.dense_hidden(inputs)
        state_embeddings = self.dense_embedding(hidden)
        return state_embeddings


class InverseModel(tf.keras.Model):
    def __init__(self, action_space, scale, **kwargs):
        super(InverseModel, self).__init__(**kwargs)

        self.action_space = action_space

        kernel_initializer = tf.initializers.variance_scaling(scale=2.0)
        logits_initializer = tf.initializers.variance_scaling(scale=1.0)

        self.dense_hidden = tf.keras.layers.Dense(
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

    def call(self,
             state_embeddings,
             next_state_embeddings,
             training=None,
             reset_state=None):
        inputs = tf.concat([state_embeddings, next_state_embeddings], axis=-1)
        hidden = self.dense_hidden(inputs)

        loc = self.dense_loc(hidden)
        loc = pynr.math.rescale(
            loc,
            oldmin=-1,
            oldmax=1,
            newmin=self.action_space.low,
            newmax=self.action_space.high)

        # TODO: distribution?
        return tfp.distributions.MultivariateNormalDiag(
            loc=loc, scale_diag=self.scale_diag)


class Policy(tf.keras.Model):
    def __init__(self, state_embedding, inverse_model, **kwargs):
        super(Policy, self).__init__(**kwargs)

        self.state_embedding = state_embedding
        self.inverse_model = inverse_model

        kernel_initializer = tf.initializers.variance_scaling(scale=2.0)

        self.dense_embedding = tf.keras.layers.Dense(
            units=64,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)

    def call(self, states, training=None, reset_state=None):
        state_embeddings = self.state_embedding(
            states, training=training, reset_state=reset_state)
        # TODO: distribution?
        goal_embeddings = self.dense_embedding(state_embeddings)
        goal_embeddings = goal_embeddings + state_embeddings  # residual
        dist = self.inverse_model(state_embeddings, goal_embeddings)
        return dist


class Value(tf.keras.Model):
    def __init__(self, state_embedding, **kwargs):
        super(Value, self).__init__(**kwargs)

        self.state_embedding = state_embedding

        logits_initializer = tf.initializers.variance_scaling(scale=1.0)

        self.dense_logits = tf.keras.layers.Dense(
            units=1, activation=None, kernel_initializer=logits_initializer)

    def call(self, inputs, training=None, reset_state=None):
        state_embeddings = self.state_embedding(
            inputs, training=training, reset_state=reset_state)
        value = self.dense_logits(state_embeddings)
        value = tf.squeeze(value, axis=-1)
        return value
