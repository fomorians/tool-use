import pyoneer as pynr
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import tensorflow_probability as tfp


class ObservationEmbedding(tf.keras.Model):

    embedding_size = 64

    def __init__(self):
        super(ObservationEmbedding, self).__init__()

        kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2.0)

        self.dense_prop = tf.keras.layers.Dense(
            units=self.embedding_size,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)
        self.dense_env = tf.keras.layers.Dense(
            units=self.embedding_size,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)
        self.dense_embed = tf.keras.layers.Dense(
            units=self.embedding_size,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)

    def call(self, observations, training=None):
        observations = tf.convert_to_tensor(observations, dtype=self.dtype)
        observations_prop = observations[..., :7 * 4]
        observations_env = observations[..., 7 * 4:]
        hidden_prop = self.dense_prop(observations_prop)
        hidden_env = self.dense_env(observations_env)
        hidden = tf.concat([hidden_prop, hidden_env], axis=-1)
        embedding = self.dense_embed(hidden)
        return embedding


class ActionEmbedding(tf.keras.Model):

    embedding_size = 64

    def __init__(self):
        super(ActionEmbedding, self).__init__()

        kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2.0)

        self.dense_embed = tf.keras.layers.Dense(
            units=self.embedding_size,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)

    def call(self, actions, training=None):
        actions = tf.convert_to_tensor(actions, dtype=self.dtype)
        embedding = self.dense_embed(actions)
        return embedding


class Forward(tf.keras.Model):

    embedding_size = 64

    def __init__(self):
        super(Forward, self).__init__()

        kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2.0)

        self.dense_hidden = tf.keras.layers.Dense(
            units=self.embedding_size,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)
        self.dense_embed = tf.keras.layers.Dense(
            units=ObservationEmbedding.embedding_size,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)

    def call(self, observations_embedding, actions_embedding, training=None):
        inputs = tf.concat(
            [observations_embedding, actions_embedding], axis=-1)
        hidden = self.dense_hidden(inputs)
        embedding = self.dense_embed(hidden)
        return embedding


class Inverse(tf.keras.Model):

    embedding_size = 64

    def __init__(self):
        super(Inverse, self).__init__()

        kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2.0)

        self.dense_hidden = tf.keras.layers.Dense(
            units=self.embedding_size,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)
        self.dense_embedding = tf.keras.layers.Dense(
            units=ActionEmbedding.embedding_size,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)

    def call(self,
             observations_embedding,
             observations_next_embedding,
             training=None):
        inputs = tf.concat(
            [observations_embedding, observations_next_embedding], axis=-1)
        hidden = self.dense_hidden(inputs)
        embedding = self.dense_embedding(hidden)
        return embedding


class Policy(tf.keras.Model):

    embedding_size = 64

    def __init__(self, action_size, scale):
        super(Policy, self).__init__()

        kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2.0)
        logits_initializer = tf.keras.initializers.VarianceScaling(scale=1.0)
        scale_initializer = pynr.initializers.SoftplusInverse(scale=scale)

        self.dense_hidden = tf.keras.layers.Dense(
            units=self.embedding_size,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)
        self.dense_loc = tf.keras.layers.Dense(
            units=action_size,
            activation=tf.math.tanh,
            kernel_initializer=logits_initializer)
        self.scale_diag_inverse = tfe.Variable(
            scale_initializer(action_size), trainable=True)

    @property
    def scale_diag(self):
        return tf.nn.softplus(self.scale_diag_inverse)

    def call(self, embedding, training=None):
        hidden = self.dense_hidden(embedding)
        loc = self.dense_loc(hidden)
        dist = tfp.distributions.MultivariateNormalDiag(
            loc=loc, scale_diag=self.scale_diag)
        return dist


class Value(tf.keras.Model):

    embedding_size = 64

    def __init__(self):
        super(Value, self).__init__()

        kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2.0)
        logits_initializer = tf.keras.initializers.VarianceScaling(scale=1.0)

        self.dense_hidden = tf.keras.layers.Dense(
            units=self.embedding_size,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)
        self.dense_values = tf.keras.layers.Dense(
            units=1, activation=None, kernel_initializer=logits_initializer)

    def call(self, embedding, training=None):
        hidden = self.dense_hidden(embedding)
        values = self.dense_values(hidden)
        return values[..., 0]


class Model(tf.keras.Model):
    def __init__(self, action_size, scale):
        super(Model, self).__init__()
        self.observation_embedding = ObservationEmbedding()
        self.action_embedding = ActionEmbedding()
        self.policy = Policy(action_size=action_size, scale=scale)
        self.value = Value()
        self.forward_model = Forward()
        self.inverse_model = Inverse()

    @tfe.defun
    def forward(self,
                observations,
                actions,
                observations_next=None,
                training=None,
                include=None):
        if include is None:
            include = ['log_probs', 'entropy', 'values', 'forward', 'inverse']

        predictions = {}

        observations_embedding = self.observation_embedding(
            observations, training=training)
        predictions['observations_embedding'] = observations_embedding

        if actions is not None:
            actions_embedding = self.action_embedding(
                actions, training=training)
            predictions['actions_embedding'] = actions_embedding

        if 'values' in include:
            predictions['values'] = self.value(
                observations_embedding, training=training)

        if 'log_probs' in include or 'entropy' in include:
            dist = self.policy(observations_embedding, training=training)

            if 'log_probs' in include:
                assert actions is not None
                predictions['log_probs'] = dist.log_prob(actions)

            if 'entropy' in include:
                predictions['entropy'] = dist.entropy()

        if observations_next is not None:
            observations_next_embedding = self.observation_embedding(
                observations_next, training=training)
            predictions[
                'observations_next_embedding'] = observations_next_embedding

        if 'forward' in include:
            predictions[
                'observations_next_embedding_pred'] = self.forward_model(
                    observations_embedding,
                    actions_embedding,
                    training=training)

        if 'inverse' in include:
            predictions['actions_embedding_pred'] = self.inverse_model(
                observations_embedding,
                observations_next_embedding,
                training=training)

        return predictions

    def call(self, observations, training=None):
        observations_embedding = self.observation_embedding(
            observations, training=training)
        dist = self.policy(observations_embedding, training=training)
        return dist
