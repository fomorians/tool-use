import os
import time
import random
import imageio
import argparse
import numpy as np
import tensorflow as tf

import pyoneer.rl as pyrl

from tool_use.env import create_env
from tool_use.model import Model
from tool_use.params import HyperParams
from tool_use.rollout import Rollout


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
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # environment
    env = create_env(args.env_name)
    env.seed(args.seed)

    # models
    model = Model(
        observation_space=env.observation_space, action_space=env.action_space
    )

    # strategies
    policy = pyrl.strategies.Mode(model)

    # checkpoints
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, directory=args.job_dir, max_to_keep=None
    )
    checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()

    # rollouts
    rollout = Rollout(env, max_episode_steps=params.max_episode_steps)

    # rollout
    transitions = rollout(policy, episodes=args.episodes, render=True)
    episodic_rewards = np.mean(np.sum(transitions["rewards"], axis=-1))
    print("episodic_rewards", episodic_rewards)

    # save
    timestamp = int(time.time())
    for episode, episode_images in enumerate(transitions["images"]):
        image_path = os.path.join(
            args.job_dir, "render_{}_{}.gif".format(timestamp, episode)
        )
        episode_weights = transitions["weights"][episode]
        max_episode_steps = int(episode_weights.sum())
        imageio.mimwrite(
            image_path,
            episode_images[: max_episode_steps + 1],
            loop=1,
            fps=2,
            subrectangles=True,
        )


if __name__ == "__main__":
    main()
