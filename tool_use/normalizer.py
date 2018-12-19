import pyoneer as pynr
import tensorflow as tf
import tensorflow.contrib.eager as tfe


class HighLowNormalizer(tf.keras.Model):
    def __init__(self, low, high, alpha, center=True, scale=True):
        super(HighLowNormalizer, self).__init__()

        self.center = center
        self.scale = scale
        self.alpha = alpha

        loc, scale = pynr.math.high_low_loc_and_scale(low=low, high=high)

        self.mean = tfe.Variable(loc, trainable=False)
        self.std = tfe.Variable(scale, trainable=False)

    def call(self, inputs, training=None):
        if training:
            new_mean, new_var = tf.nn.moments(inputs, axes=[0, 1])
            new_std = tf.sqrt(new_var)

            self.mean.assign(self.mean * self.alpha +
                             new_mean * (1 - self.alpha))
            self.std.assign(self.std * self.alpha + new_std * (1 - self.alpha))

        if self.center:
            inputs = inputs - self.mean[None, None, ...]

        if self.scale:
            inputs = pynr.math.safe_divide(inputs, self.std[None, None, ...])

        inputs = tf.check_numerics(inputs, 'inputs')
        return inputs


class Normalizer(tf.keras.Model):
    def __init__(self, shape, center=True, scale=True):
        super(Normalizer, self).__init__()

        self.center = center
        self.scale = scale

        self.count = tfe.Variable(0, dtype=tf.int32, trainable=False)
        self.mean = tfe.Variable(
            tf.zeros(shape=shape, dtype=tf.float32), trainable=False)
        self.var_sum = tfe.Variable(
            tf.zeros(shape=shape, dtype=tf.float32), trainable=False)

    @property
    def std(self):
        return tf.sqrt(
            tf.maximum(self.var_sum / tf.to_float(self.count - 1), 0))

    def call(self, inputs, training=None):
        if training:
            self.count.assign_add(tf.reduce_prod(inputs.shape[:2]))

            mean_deltas = tf.reduce_sum(
                inputs - self.mean[None, None, ...], axis=(0, 1))
            new_mean = tf.where(
                tf.greater(self.count, 1),
                self.mean + (mean_deltas / tf.to_float(self.count)),
                inputs[0, 0])

            var_deltas = (inputs - self.mean[None, None, ...]) * (
                inputs - new_mean[None, None, ...])
            new_var_sum = self.var_sum + tf.reduce_sum(var_deltas, axis=(0, 1))

            self.mean.assign(new_mean)
            self.var_sum.assign(new_var_sum)

        if self.center:
            inputs = inputs - self.mean[None, None, ...]

        if self.scale:
            inputs = pynr.math.safe_divide(inputs, self.std[None, None, ...])

        inputs = tf.check_numerics(inputs, 'inputs')
        return inputs
