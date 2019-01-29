import gym
import random
import argparse
import numpy as np
import tensorflow as tf
import pyoneer.rl as pyrl

from tool_use import models
from tool_use.params import HyperParams
from tool_use.rollout import Rollout
from tool_use.wrappers import RangeNormalize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', required=True)
    parser.add_argument('--env', default='Pendulum-v0')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--episodes', default=10, type=int)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    # params
    params = HyperParams(env=args.env, seed=args.seed)

    # eager
    tf.enable_eager_execution()

    # environment
    env = gym.make(args.env)
    env = RangeNormalize(env)

    observation_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    # seeding
    env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # policies
    policy = models.PolicyValue(
        action_size=env.action_space.shape[0], scale=params.scale)

    mock_observations = tf.zeros(
        shape=(1, 1, observation_size), dtype=np.float32)
    mock_actions = tf.zeros(shape=(1, 1, action_size), dtype=np.float32)
    policy.forward(mock_observations, mock_actions, reset_state=True)

    # checkpoints
    checkpoint = tf.train.Checkpoint(policy=policy)
    checkpoint_path = tf.train.latest_checkpoint(args.job_dir)
    assert checkpoint_path is not None
    checkpoint.restore(checkpoint_path)

    # rollouts
    rollout = Rollout(env, max_episode_steps=env.spec.max_episode_steps)

    # strategies
    inference_strategy = pyrl.strategies.ModeStrategy(policy)

    # rollouts
    render = (args.render and args.env != 'KukaEnv-v0')
    observations, actions, rewards, observations_next, weights = rollout(
        inference_strategy, episodes=args.episodes, render=render)
    episodic_reward = tf.reduce_mean(tf.reduce_sum(rewards, axis=-1))
    print('episodic_reward:', episodic_reward.numpy())


if __name__ == '__main__':
    main()
