import tensorflow as tf
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
        log_probs -= tf.reduce_sum(tf.log(1 - value**2 + 1e-6), axis=-1)
        return log_probs
