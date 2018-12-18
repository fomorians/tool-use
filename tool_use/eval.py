import random
import argparse
import numpy as np
import tensorflow as tf
import pyoneer.rl as pyrl

# from tool_use.env import KukaEnv
from tool_use.models import Policy
from tool_use.params import HyperParams
from tool_use.rollout import Rollout

from gym.envs.classic_control import PendulumEnv


class PolicyWrapper:
    def __init__(self, policy):
        self.policy = policy

    def __call__(self, state, *args, **kwargs):
        state = tf.convert_to_tensor(state, dtype=np.float32)
        state_batch = state[None, None, ...]
        action_batch = self.policy(state_batch, *args, **kwargs)
        action = action_batch[0, 0]
        action = action.numpy()
        return action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', required=True)
    parser.add_argument('--seed', default=42)
    parser.add_argument('--episodes', default=10)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    tf.enable_eager_execution()

    params = HyperParams()

    # env = KukaEnv(render=args.render)
    env = PendulumEnv()

    env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    rollout = Rollout(env, max_episode_steps=params.max_episode_steps)

    policy = Policy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        scale=params.scale)

    inference_strategy = pyrl.strategies.ModeStrategy(policy)
    inference_strategy = PolicyWrapper(inference_strategy)

    checkpoint = tf.train.Checkpoint(policy=policy)
    checkpoint_path = tf.train.latest_checkpoint(args.job_dir)
    assert checkpoint_path is not None
    checkpoint.restore(checkpoint_path)

    states, actions, rewards, next_states, weights = rollout(
        inference_strategy, episodes=args.episodes, render=args.render)
    episodic_reward = tf.reduce_mean(tf.reduce_sum(rewards, axis=-1))
    print('episodic_reward:', episodic_reward)


if __name__ == '__main__':
    main()
