import gym
import pyoneer.rl as pyrl


def create_env(env_name, seed):
    env = gym.make(env_name)
    env = pyrl.wrappers.ObservationCoordinates(env)
    env = pyrl.wrappers.ObservationNormalization(env)
    env.seed(seed)
    return env