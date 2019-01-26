import pyoneer as pynr
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import tensorflow_probability as tfp


class StateEmbedding(tf.keras.Model):
    def __init__(self, trainable=True, **kwargs):
        super(StateEmbedding, self).__init__(**kwargs)
        kernel_initializer = tf.keras.initializers.VarianceScaling()
        self.dense_embedding = tf.keras.layers.Dense(
            units=64,
            activation=None,
            kernel_initializer=kernel_initializer,
            trainable=trainable)

    def call(self, inputs, training=None, reset_state=None):
        return self.dense_embedding(inputs)


class ActionEmbedding(tf.keras.Model):
    def __init__(self, **kwargs):
        super(ActionEmbedding, self).__init__(**kwargs)
        kernel_initializer = tf.keras.initializers.VarianceScaling()
        self.dense_embedding = tf.keras.layers.Dense(
            units=64, activation=None, kernel_initializer=kernel_initializer)

    def call(self, inputs, training=None, reset_state=None):
        return self.dense_embedding(inputs)


class ForwardModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(ForwardModel, self).__init__(**kwargs)
        kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2.0)
        logits_initializer = tf.keras.initializers.VarianceScaling()
        self.dense_hidden = tf.keras.layers.Dense(
            units=64,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)
        self.dense_logits = tf.keras.layers.Dense(
            units=64, activation=None, kernel_initializer=logits_initializer)

    def call(self, states, actions, training=None, reset_state=None):
        inputs = tf.concat([states, actions], axis=-1)
        hidden = self.dense_hidden(inputs)
        logits = self.dense_logits(hidden)
        return logits


class InverseModel(tf.keras.Model):
    def __init__(self, action_size, trainable=True, **kwargs):
        super(InverseModel, self).__init__(**kwargs)
        self.action_size = action_size
        kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2.0)
        logits_initializer = tf.keras.initializers.VarianceScaling()
        self.dense_hidden = tf.keras.layers.Dense(
            units=64,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer,
            trainable=trainable)
        self.dense_logits = tf.keras.layers.Dense(
            units=action_size,
            activation=None,
            kernel_initializer=logits_initializer,
            trainable=trainable)

    def call(self, states, next_states, training=None, reset_state=None):
        inputs = tf.concat([states, next_states], axis=-1)
        hidden = self.dense_hidden(inputs)
        logits = self.dense_logits(hidden)
        return logits


class Policy(tf.keras.Model):
    def __init__(self, action_size, scale, **kwargs):
        super(Policy, self).__init__(**kwargs)

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
        self.dense_loc = tf.keras.layers.Dense(
            units=action_size,
            activation=None,
            kernel_initializer=logits_initializer)

        self.scale_diag_inverse = tfe.Variable(
            scale_initializer(action_size), trainable=True)

    @property
    def scale_diag(self):
        return tf.nn.softplus(self.scale_diag_inverse)

    def call(self, inputs, training=None, reset_state=None):
        inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
        hidden = self.dense_hidden1(inputs)
        hidden = self.dense_hidden2(hidden)
        loc = self.dense_loc(hidden)
        dist = tfp.distributions.MultivariateNormalDiag(
            loc=loc, scale_diag=self.scale_diag)
        return dist


class Value(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Value, self).__init__(**kwargs)

        kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2.0)
        logits_initializer = tf.keras.initializers.VarianceScaling(scale=1.0)

        self.dense_hidden1 = tf.keras.layers.Dense(
            units=64,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)
        self.dense_hidden2 = tf.keras.layers.Dense(
            units=64,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)
        self.dense_value = tf.keras.layers.Dense(
            units=1, activation=None, kernel_initializer=logits_initializer)

    def call(self, inputs, training=None, reset_state=None):
        inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
        hidden = self.dense_hidden1(inputs)
        hidden = self.dense_hidden2(hidden)
        value = self.dense_value(hidden)
        return value[..., 0]
