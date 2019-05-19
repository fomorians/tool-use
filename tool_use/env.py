import gym
import numpy as np
import pyoneer.rl as pyrl


class RandomRewards(gym.Wrapper):
    def __init__(self, env, probability):
        super(RandomRewards, self).__init__(env)
        self.probability = probability

    def sample_reward(self):
        reward = float(np.random.binomial(1, self.probability))
        return reward

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward = self.sample_reward()
        return observation, reward, done, info


def create_env(env_name, seed):
    env = gym.make(env_name)
    env = pyrl.wrappers.ObservationCoordinates(env)
    env = pyrl.wrappers.ObservationNormalization(env)
    env.seed(seed)
    return env
