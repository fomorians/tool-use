import os
import gym
import time
import random
import imageio
import argparse
import numpy as np
import tensorflow as tf

import pyoneer.rl as pyrl

from tool_use.models import Model
from tool_use.params import HyperParams
from tool_use.rollouts import Rollout
from tool_use.wrappers import ObservationCoordinates, ObservationNormalization


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--episodes", default=1, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    print(args)

    # params
    params_path = os.path.join(args.job_dir, "params.json")
    params = HyperParams.load(params_path)
    print(params)

    # seed
    random.seed(params.seed)
    np.random.seed(params.seed)
    tf.random.set_seed(params.seed)

    # environment
    env = gym.make(args.env_name)
    env = ObservationCoordinates(env)
    env = ObservationNormalization(env)
    env.seed(args.seed)

    print(env.resize_scale)

    # models
    model = Model(action_space=env.action_space)

    # strategies
    inference_strategy = pyrl.strategies.ModeStrategy(model.get_distribution)

    # checkpoints
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint_path = tf.train.latest_checkpoint(args.job_dir)
    assert checkpoint_path is not None
    checkpoint.restore(checkpoint_path)

    # rollouts
    rollout = Rollout(env, max_episode_steps=params.max_episode_steps)

    # rollout
    transitions = rollout(inference_strategy, env, episodes=args.episodes, render=True)
    episodic_rewards = tf.reduce_mean(tf.reduce_sum(transitions["rewards"], axis=-1))
    tf.print("episodic_rewards", episodic_rewards)

    # save
    timestamp = int(time.time())
    for episode, episode_images in enumerate(transitions["images"]):
        image_path = os.path.join(
            args.job_dir, "render_{}_{}.gif".format(timestamp, episode)
        )
        imageio.mimwrite(image_path, episode_images, loop=1, fps=5, subrectangles=True)


if __name__ == "__main__":
    main()
