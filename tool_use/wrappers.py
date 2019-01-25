import gym
import numpy as np


class RangeNormalize:
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def observation_space(self):
        observation_space = self.env.observation_space
        low = -np.ones(
            shape=observation_space.shape, dtype=observation_space.dtype)
        high = np.ones(
            shape=observation_space.shape, dtype=observation_space.dtype)
        return gym.spaces.Box(low, high, dtype=np.float32)

    @property
    def action_space(self):
        action_space = self.env.action_space
        low = -np.ones(shape=action_space.shape, dtype=action_space.dtype)
        high = np.ones(shape=action_space.shape, dtype=action_space.dtype)
        return gym.spaces.Box(low, high, dtype=np.float32)

    def denormalize_action(self, action):
        low = self.env.action_space.low
        high = self.env.action_space.high
        action = (action + 1) / 2 * (high - low) + low
        return action

    def normalize_observation(self, observ):
        low = self.env.observation_space.low
        high = self.env.observation_space.high
        observ = 2 * (observ - low) / (high - low) - 1
        return observ

    def reset(self):
        return self.normalize_observation(self.env.reset())

    def step(self, action):
        action = self.denormalize_action(action)
        observation, reward, done, info = self.env.step(action)
        observation = self.normalize_observation(observation)
        return observation, reward, done, info
