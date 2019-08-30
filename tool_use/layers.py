import pyoneer as pynr
import tensorflow as tf


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)

        kernel_initializer = tf.initializers.VarianceScaling(scale=2.0)

        self.conv1 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=1,
            padding="same",
            activation=None,
            kernel_initializer=kernel_initializer,
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=1,
            padding="same",
            activation=None,
            kernel_initializer=kernel_initializer,
        )

    def call(self, inputs):
        hidden = pynr.activations.swish(inputs)
        hidden = self.conv1(hidden)
        hidden = pynr.activations.swish(hidden)
        hidden = self.conv2(hidden)
        hidden += inputs
        return hidden
