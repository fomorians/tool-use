import gym
import numpy as np
import pyoneer as pynr
import tensorflow as tf
import tensorflow_probability as tfp


class Policy(tf.keras.Model):
    def __init__(self, action_space, **kwargs):
        super(Policy, self).__init__(**kwargs)

        self.action_space = action_space
        self.is_discrete = isinstance(action_space, gym.spaces.Discrete)

        if self.is_discrete:
            self.num_outputs = action_space.n
        else:
            self.num_outputs = action_space.shape[-1]

        kernel_initializer = tf.initializers.VarianceScaling(scale=2.0)
        logits_initializer = tf.initializers.VarianceScaling(scale=1.0)

        self.hidden1 = tf.keras.layers.Dense(
            units=64,
            activation=pynr.activations.swish,
            kernel_initializer=kernel_initializer,
        )
        self.hidden2 = tf.keras.layers.Dense(
            units=64,
            activation=pynr.activations.swish,
            kernel_initializer=kernel_initializer,
        )

        if self.is_discrete:
            self.logits = tf.keras.layers.Dense(
                units=self.num_outputs,
                activation=None,
                kernel_initializer=logits_initializer,
            )
        else:
            self.loc = tf.keras.layers.Dense(
                units=self.num_outputs,
                activation=None,
                kernel_initializer=logits_initializer,
            )
            self.scale_diag = tf.keras.layers.Dense(
                units=self.num_outputs,
                activation=tf.exp,
                kernel_initializer=logits_initializer,
            )

    def _discrete(self, hidden):
        logits = self.logits(hidden)
        dist = tfp.distributions.RelaxedOneHotCategorical(
            temperature=1.0, logits=logits
        )
        return dist

    def _continuous(self, hidden):
        loc = self.loc(hidden)
        scale_diag = self.scale_diag(hidden)

        bijector = tfp.bijectors.Chain(
            [
                tfp.bijectors.Tanh(),
                tfp.bijectors.Affine(shift=loc, scale_diag=scale_diag),
            ]
        )

        base_dist = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros_like(loc), scale_diag=tf.ones_like(scale_diag)
        )
        dist = tfp.distributions.TransformedDistribution(
            distribution=base_dist, bijector=bijector
        )
        return dist

    def sample(self, *args, **kwargs):
        dist = self.call(*args, **kwargs)
        if self.is_discrete:
            actions = dist.sample()
        else:
            actions = dist.bijector.forward(dist.distribution.sample())
        return actions

    def mode(self, *args, **kwargs):
        dist = self.call(*args, **kwargs)
        if self.is_discrete:
            actions = tf.one_hot(tf.argmax(dist.probs, axis=-1), depth=self.num_outputs)
        else:
            actions = dist.bijector.forward(dist.distribution.mode())
        return actions

    def call(self, observations, training=None, reset_state=None):
        hidden = self.hidden1(observations)
        hidden = self.hidden2(hidden)

        if self.is_discrete:
            dist = self._discrete(hidden)
        else:
            dist = self._continuous(hidden)
        return dist


class QFunction(tf.keras.Model):
    def __init__(self, **kwargs):
        super(QFunction, self).__init__(**kwargs)

        kernel_initializer = tf.initializers.VarianceScaling(scale=2.0)
        logits_initializer = tf.initializers.VarianceScaling(scale=1.0)

        self.hidden1 = tf.keras.layers.Dense(
            units=64,
            activation=pynr.activations.swish,
            kernel_initializer=kernel_initializer,
        )
        self.hidden2 = tf.keras.layers.Dense(
            units=64,
            activation=pynr.activations.swish,
            kernel_initializer=kernel_initializer,
        )
        self.logits = tf.keras.layers.Dense(
            units=1, activation=None, kernel_initializer=logits_initializer
        )

    def call(self, inputs, training=None, reset_state=None):
        observations = inputs["observations"]
        actions = inputs["actions"]

        hidden = self.hidden1(observations)
        hidden = self.hidden2(tf.concat([hidden, actions], axis=-1))
        logits = tf.squeeze(self.logits(hidden), axis=-1)

        return logits
