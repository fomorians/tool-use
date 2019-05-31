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
from tool_use.batch_rollout import BatchRollout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--episodes", default=128, type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    print(args)

    timestamp = int(time.time())
    image_dir = os.path.join(args.job_dir, "results", args.env_name, str(timestamp))
    success_image_dir = os.path.join(image_dir, "success")
    failure_image_dir = os.path.join(image_dir, "failure")

    # make image directories
    os.makedirs(success_image_dir, exist_ok=True)
    os.makedirs(failure_image_dir, exist_ok=True)

    # params
    params_path = os.path.join(args.job_dir, "params.json")
    params = HyperParams.load(params_path)
    print(params)

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # environment
    env = pyrl.wrappers.Batch(
        lambda batch_id: create_env(args.env_name), batch_size=params.env_batch_size
    )
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
    rollout = BatchRollout(env, max_episode_steps=params.max_episode_steps)

    # rollout
    transitions = rollout(policy, episodes=args.episodes, render=True)
    episodic_rewards = np.mean(np.sum(transitions["rewards"], axis=-1))
    print("episodic_rewards", episodic_rewards)

    # save
    for episode, episode_images in enumerate(transitions["images"]):
        rewards = np.sum(transitions["rewards"][episode], axis=-1)

        if rewards > 0:
            image_path = os.path.join(success_image_dir, "{}.gif".format(episode))
        else:
            image_path = os.path.join(failure_image_dir, "{}.gif".format(episode))

        episode_weights = transitions["weights"][episode]
        max_episode_steps = int(episode_weights.sum())
        imageio.mimwrite(
            image_path,
            episode_images[: max_episode_steps + 1],
            loop=1,
            fps=3,
            subrectangles=True,
        )


if __name__ == "__main__":
    main()
