import numpy as np
import pyoneer as pynr
import tensorflow as tf

from tool_use.noise import OrnsteinUhlenbeckNoise


class RandomStrategy:
    def __init__(self, action_space):
        self.action_space = action_space

        loc, var = pynr.nn.moments_from_range(
            minval=action_space.low, maxval=action_space.high)
        scale = np.sqrt(var)
        self.ou_process = OrnsteinUhlenbeckNoise(loc, scale)

    def __call__(self, *args, **kwargs):
        action = self.ou_process.sample()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return tf.convert_to_tensor(action[None, None, ...])
