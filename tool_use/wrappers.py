import gym
import numpy as np


def one_hot(x, depth):
    return np.eye(depth)[x]


class OneHotObservation:
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def observation_space(self):
        observation_shape = (self.env.observation_space.n,)
        observation_dtype = np.float32
        low = np.zeros(shape=observation_shape, dtype=observation_dtype)
        high = np.ones(shape=observation_shape, dtype=observation_dtype)
        return gym.spaces.Box(low, high, dtype=np.float32)

    def expand_observation(self, observation):
        observation_one_hot = one_hot(observation, self.env.observation_space.n)
        return observation_one_hot

    def reset(self):
        return self.expand_observation(self.env.reset())

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = self.expand_observation(observation)
        return observation, reward, done, info


class ObservationCoordinates:
    def __init__(self, env):
        self.env = env
        self.board_size = env.unwrapped.nrow * env.unwrapped.ncol

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def observation_space(self):
        observation_shape = self.env.observation_space.shape + (2,)
        observation_dtype = np.float32
        low = np.zeros(shape=observation_shape, dtype=observation_dtype)
        high = np.ones(shape=observation_shape, dtype=observation_dtype)
        space = gym.spaces.Box(low, high, dtype=np.float32)
        return space

    def concat_coords(self, observation):
        coords = np.linspace(start=0, stop=1, num=self.board_size)
        observation = np.stack([observation, coords], axis=-1)
        return observation

    def reset(self):
        return self.concat_coords(self.env.reset())

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = self.concat_coords(observation)
        return observation, reward, done, info


class ObservationNormalize:
    def __init__(self, env):
        assert isinstance(env.observation_space, gym.spaces.Box)
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def observation_space(self):
        observation_space = self.env.observation_space
        low = -np.ones(shape=observation_space.shape, dtype=observation_space.dtype)
        high = np.ones(shape=observation_space.shape, dtype=observation_space.dtype)
        return gym.spaces.Box(low, high, dtype=np.float32)

    def normalize_observation(self, observ):
        low = self.env.observation_space.low
        high = self.env.observation_space.high
        observ = 2 * (observ - low) / (high - low) - 1
        return observ

    def reset(self):
        return self.normalize_observation(self.env.reset())

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = self.normalize_observation(observation)
        return observation, reward, done, info


class ActionNormalize:
    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Box)
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

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

    def step(self, action):
        action = self.denormalize_action(action)
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, info
