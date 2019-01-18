import gym
import random
import argparse
import numpy as np
import tensorflow as tf
import pyoneer.rl as pyrl

from tool_use.models import Policy
from tool_use.params import HyperParams
from tool_use.rollout import Rollout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', required=True)
    parser.add_argument('--env', default='Pendulum-v0')
    parser.add_argument('--seed', default=42)
    parser.add_argument('--episodes', default=10)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    # register kuka env
    gym.envs.register(
        id='KukaEnv-v0',
        entry_point='tool_use.kuka_env:KukaEnv',
        max_episode_steps=100,
        kwargs=dict(render=args.render))

    # params
    params = HyperParams(env=args.env, seed=args.seed)

    # eager
    tf.enable_eager_execution()

    # environment
    env = gym.make(args.env)

    # seeding
    env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # rollouts
    rollout = Rollout(env, max_episode_steps=env.spec.max_episode_steps)

    # policies
    policy = Policy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        scale=params.scale)
    inference_strategy = pyrl.strategies.ModeStrategy(policy)

    # checkpoints
    checkpoint = tf.train.Checkpoint(policy=policy)
    checkpoint_path = tf.train.latest_checkpoint(args.job_dir)
    assert checkpoint_path is not None
    checkpoint.restore(checkpoint_path)

    # rollouts
    states, actions, rewards, next_states, weights = rollout(
        inference_strategy, episodes=args.episodes, render=args.render)
    episodic_reward = tf.reduce_mean(tf.reduce_sum(rewards, axis=-1))
    print('episodic_reward:', episodic_reward)


if __name__ == '__main__':
    main()
