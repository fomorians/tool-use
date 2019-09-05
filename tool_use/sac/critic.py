import pyoneer as pynr
import tensorflow as tf
import tensorflow_probability as tfp

from tool_use.layers import ResidualBlock


class Critic(tf.keras.Model):
    def __init__(self, batch_size, **kwargs):
        super(Critic, self).__init__(**kwargs)

        kernel_initializer = tf.initializers.VarianceScaling(scale=2.0)
        logits_initializer = tf.initializers.VarianceScaling(scale=1.0)

        self.state = tf.Variable(tf.zeros(shape=[batch_size, 64]), trainable=False)

        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=2,
            strides=1,
            padding="same",
            activation=pynr.activations.swish,
            kernel_initializer=kernel_initializer,
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=2,
            strides=1,
            padding="same",
            activation=pynr.activations.swish,
            kernel_initializer=kernel_initializer,
        )

        self.downsample1 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
        self.residual_block1 = ResidualBlock(filters=64)
        self.downsample2 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
        self.residual_block2 = ResidualBlock(filters=64)

        self.global_pool = tf.keras.layers.GlobalMaxPool2D()

        self.dense_hidden = tf.keras.layers.Dense(
            units=64,
            activation=pynr.activations.swish,
            kernel_initializer=kernel_initializer,
        )

        self.rnn = tf.keras.layers.GRU(
            units=64, return_sequences=True, return_state=True
        )

        self.logits = tf.keras.layers.Dense(
            units=1, activation=None, kernel_initializer=logits_initializer
        )

    @tf.function
    def call(self, inputs, training=None, reset_state=None):
        observations = inputs["observations"]
        actions = inputs["actions"]

        batch_size, steps, height, width, channels = observations.shape
        action_size, action_dim = actions.shape[2:]

        observations = tf.reshape(
            observations, [batch_size * steps, height, width, channels]
        )
        actions = tf.reshape(actions, [batch_size * steps, action_size * action_dim])

        hidden = self.conv1(observations)
        hidden = self.conv2(hidden)

        hidden = self.downsample1(hidden)
        hidden = self.residual_block1(hidden)
        hidden = self.downsample2(hidden)
        hidden = self.residual_block2(hidden)

        hidden = self.global_pool(hidden)

        hidden = self.dense_hidden(tf.concat([hidden, actions], axis=-1))

        hidden = tf.reshape(hidden, [batch_size, steps, self.dense_hidden.units])

        if reset_state:
            self.state.assign(tf.zeros_like(self.state))
        hidden, state = self.rnn(hidden, initial_state=self.state)
        self.state.assign(state)

        logits = tf.squeeze(self.logits(hidden), axis=-1)

        return logits
