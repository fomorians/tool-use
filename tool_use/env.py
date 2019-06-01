import gym
import tensorflow as tf
import pyoneer.rl as pyrl

from tool_use.batch_rollout import BatchRollout


def create_env(env_name):
    env = gym.make(env_name)
    env = pyrl.wrappers.ObservationCoordinates(env)
    env = pyrl.wrappers.ObservationNormalization(env)
    return env


def collect_transitions(env_name, episodes, policy, seed, params, render=False):
    with tf.device("/cpu:0"):
        env = pyrl.wrappers.Batch(
            lambda batch_id: create_env(env_name), batch_size=params.env_batch_size
        )
        env.seed(seed)
        rollout = BatchRollout(env, params.max_episode_steps)
        transitions = rollout(policy, episodes, render=render)
    return transitions
