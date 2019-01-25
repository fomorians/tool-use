import numpy as np
import pyoneer as pynr
import tensorflow as tf


class OrnsteinUhlenbeckNoise:
    def __init__(self, action_space, friction=0.15, dt=1e-2):
        self.action_space = action_space
        self.loc, var = pynr.nn.moments_from_range(
            minval=action_space.low, maxval=action_space.high)
        self.scale = tf.sqrt(var)
        self.friction = friction
        self.dt = dt
        self.x = np.zeros_like(self.loc)

    def sample(self):
        delta_mean = self.friction * (self.loc - self.x) * self.dt
        brownian_velocity = np.random.normal(
            scale=self.scale, size=self.loc.shape) * np.sqrt(self.dt)
        self.x += delta_mean + brownian_velocity
        self.x = np.clip(self.x, self.action_space.low, self.action_space.high)
        return self.x


class RandomStrategy:
    def __init__(self, action_space):
        self.action_space = action_space
        self.ou_process = OrnsteinUhlenbeckNoise(action_space)

    def __call__(self, *args, **kwargs):
        action = self.ou_process.sample()
        return tf.convert_to_tensor(action[None, None, ...])
