import pyoneer as pynr
import tensorflow as tf
import tensorflow_probability as tfp


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)

        kernel_initializer = tf.initializers.VarianceScaling(scale=2.0)

        self.conv1 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=3,
            strides=1,
            padding="same",
            activation=None,
            use_bias=True,
            kernel_initializer=kernel_initializer,
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=3,
            strides=1,
            padding="same",
            activation=None,
            use_bias=True,
            kernel_initializer=kernel_initializer,
        )

    def call(self, inputs):
        hidden = pynr.nn.swish(inputs)
        hidden = self.conv1(hidden)
        hidden = pynr.nn.swish(hidden)
        hidden = self.conv2(hidden)
        hidden += inputs
        return hidden


class Model(tf.keras.Model):
    def __init__(self, observation_space, action_space):
        super(Model, self).__init__()

        self.observation_space = observation_space
        self.action_space = action_space

        kernel_initializer = tf.initializers.VarianceScaling(scale=2.0)
        logits_initializer = tf.initializers.VarianceScaling(scale=1.0)

        self.initial_hidden_state = tf.Variable(tf.zeros(shape=[1, 64]), trainable=True)
        self.initial_cell_state = tf.Variable(tf.zeros(shape=[1, 64]), trainable=True)

        self.hidden_state = None
        self.cell_state = None

        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=2,
            strides=1,
            padding="same",
            activation=pynr.nn.swish,
            use_bias=True,
            kernel_initializer=kernel_initializer,
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=2,
            strides=1,
            padding="same",
            activation=pynr.nn.swish,
            use_bias=True,
            kernel_initializer=kernel_initializer,
        )
        self.downsample1 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
        self.block1 = ResidualBlock(filters=64)
        self.downsample2 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
        self.block2 = ResidualBlock(filters=64)
        self.global_pool = tf.keras.layers.GlobalMaxPool2D()

        self.move_embedding = tf.keras.layers.Embedding(
            input_dim=action_space.nvec[0], output_dim=8
        )
        self.grasp_embedding = tf.keras.layers.Embedding(
            input_dim=action_space.nvec[1], output_dim=8
        )
        self.reward_embedding = tf.keras.layers.Dense(
            units=8,
            activation=pynr.nn.swish,
            use_bias=True,
            kernel_initializer=kernel_initializer,
        )
        self.concat = tf.keras.layers.Concatenate()

        self.dense_hidden = tf.keras.layers.Dense(
            units=64,
            activation=pynr.nn.swish,
            use_bias=True,
            kernel_initializer=kernel_initializer,
        )

        self.rnn = tf.keras.layers.LSTM(
            units=64, return_sequences=True, return_state=True
        )

        self.move_logits = tf.keras.layers.Dense(
            units=action_space.nvec[0],
            activation=None,
            kernel_initializer=logits_initializer,
        )
        self.grasp_logits = tf.keras.layers.Dense(
            units=action_space.nvec[1],
            activation=None,
            kernel_initializer=logits_initializer,
        )
        self.dense_values_logits = tf.keras.layers.Dense(
            units=1, activation=None, kernel_initializer=logits_initializer
        )

        self.dense_forward = tf.keras.layers.Dense(
            units=64,
            activation=pynr.nn.swish,
            use_bias=True,
            kernel_initializer=kernel_initializer,
        )
        self.dense_inverse = tf.keras.layers.Dense(
            units=64,
            activation=pynr.nn.swish,
            use_bias=True,
            kernel_initializer=kernel_initializer,
        )
        self.dense_inverse_move = tf.keras.layers.Dense(
            units=action_space.nvec[0],
            activation=tf.nn.softmax,
            use_bias=True,
            kernel_initializer=kernel_initializer,
        )
        self.dense_inverse_grasp = tf.keras.layers.Dense(
            units=action_space.nvec[1],
            activation=tf.nn.softmax,
            use_bias=True,
            kernel_initializer=kernel_initializer,
        )

    def _get_embedding(
        self, observations, actions_prev, rewards_prev, training=None, reset_state=None
    ):
        batch_size, steps, height, width, channels = observations.shape

        observations = tf.reshape(
            observations, [batch_size * steps, height, width, channels]
        )
        actions_prev = tf.reshape(
            actions_prev, [batch_size * steps, actions_prev.shape[-1]]
        )
        rewards_prev = tf.reshape(rewards_prev, [batch_size * steps, 1])

        hidden = self.conv1(observations)
        hidden = self.conv2(hidden)

        hidden = self.downsample1(hidden)
        hidden = self.block1(hidden)
        hidden = self.downsample2(hidden)
        hidden = self.block2(hidden)

        hidden = self.global_pool(hidden)

        move_embedding = self.move_embedding(actions_prev[..., 0])
        grasp_embedding = self.grasp_embedding(actions_prev[..., 1])
        reward_embedding = self.reward_embedding(rewards_prev)

        hidden = self.concat(
            [hidden, move_embedding, grasp_embedding, reward_embedding]
        )

        hidden = self.dense_hidden(hidden)

        hidden = tf.reshape(hidden, [batch_size, steps, self.dense_hidden.units])

        if self.hidden_state is None or self.cell_state is None or reset_state:
            self.hidden_state = tf.tile(self.initial_hidden_state, [batch_size, 1])
            self.cell_state = tf.tile(self.initial_cell_state, [batch_size, 1])

        hidden, self.hidden_state, self.cell_state = self.rnn(
            hidden, initial_state=(self.hidden_state, self.cell_state)
        )
        return hidden

    def _forward_model(self, embedding, actions):
        move_embedding = self.move_embedding(actions[..., 0])
        grasp_embedding = self.grasp_embedding(actions[..., 1])
        hidden = tf.concat([embedding, move_embedding, grasp_embedding], axis=-1)
        embedding_next = embedding + self.dense_forward(hidden)
        return embedding_next

    def _inverse_model(self, embedding, embedding_next):
        hidden = tf.concat([embedding, embedding_next], axis=-1)
        hidden = self.dense_inverse(hidden)
        move_pred = self.dense_inverse_move(hidden)
        grasp_pred = self.dense_inverse_grasp(hidden)
        return move_pred, grasp_pred

    def get_training_outputs(self, inputs, training=None, reset_state=None):
        embedding = self._get_embedding(
            observations=inputs["observations"],
            actions_prev=inputs["actions_prev"],
            rewards_prev=inputs["rewards_prev"],
            training=training,
            reset_state=reset_state,
        )
        embedding_next = self._get_embedding(
            observations=inputs["observations_next"],
            actions_prev=inputs["actions"],
            rewards_prev=inputs["rewards"],
            training=training,
            reset_state=reset_state,
        )
        embedding_next_pred = self._forward_model(embedding, inputs["actions"])
        move_pred, grasp_pred = self._inverse_model(embedding, embedding_next)

        move_logits = self.move_logits(embedding)
        grasp_logits = self.grasp_logits(embedding)
        values_logits = self.dense_values_logits(embedding)

        move_dist = tfp.distributions.Categorical(logits=move_logits)
        grasp_dist = tfp.distributions.Categorical(logits=grasp_logits)
        dist = pynr.distributions.MultiCategorical([move_dist, grasp_dist])

        log_probs = dist.log_prob(inputs["actions"])
        entropy = dist.entropy()
        values = values_logits[..., 0]

        outputs = {
            "log_probs": log_probs,
            "entropy": entropy,
            "values": values,
            "embedding_next": embedding_next,
            "embedding_next_pred": embedding_next_pred,
            "move_pred": move_pred,
            "grasp_pred": grasp_pred,
        }

        return outputs

    def call(
        self, observations, actions_prev, rewards_prev, training=None, reset_state=None
    ):
        embedding = self._get_embedding(
            observations=observations,
            actions_prev=actions_prev,
            rewards_prev=rewards_prev,
            training=training,
            reset_state=reset_state,
        )

        move_logits = self.move_logits(embedding)
        grasp_logits = self.grasp_logits(embedding)

        move_dist = tfp.distributions.Categorical(logits=move_logits)
        grasp_dist = tfp.distributions.Categorical(logits=grasp_logits)
        dist = pynr.distributions.MultiCategorical([move_dist, grasp_dist])

        return dist
