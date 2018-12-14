import random
import argparse
import numpy as np
import tensorflow as tf
import pyoneer.rl as pyrl

from gym.wrappers import TimeLimit

from tool_use.env import KukaEnv
from tool_use.models import Policy
from tool_use.params import HyperParams
from tool_use.rollout import Rollout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', required=True)
    parser.add_argument('--seed', default=42)
    parser.add_argument('--episodes', default=10)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    tf.enable_eager_execution()

    params = HyperParams()

    env = KukaEnv(render=args.render)
    env = TimeLimit(env, max_episode_steps=params.max_episode_steps)

    env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    rollout = Rollout(env, max_episode_steps=params.max_episode_steps)

    action_size = env.action_space.shape[0]

    behavioral_policy = Policy(action_size=action_size)
    policy = Policy(action_size=action_size)

    inference_strategy = pyrl.strategies.ModeStrategy(behavioral_policy)

    checkpoint = tf.train.Checkpoint(
        policy=policy, behavioral_policy=behavioral_policy)
    checkpoint_path = tf.train.latest_checkpoint(args.job_dir)
    assert checkpoint_path is not None
    checkpoint.restore(checkpoint_path)

    states, actions, rewards, next_states, weights = rollout(
        inference_strategy, episodes=args.episodes)
    episodic_reward = tf.reduce_mean(tf.reduce_sum(rewards, axis=-1))
    print('episodic_reward:', episodic_reward)


if __name__ == '__main__':
    main()
