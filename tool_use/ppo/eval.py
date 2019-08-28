import os
import time
import random
import imageio
import argparse
import numpy as np
import tensorflow as tf

import pyoneer.rl as pyrl

from tool_use import constants
from tool_use.env import create_env, collect_transitions
from tool_use.ppo.model import Model
from tool_use.ppo.params import HyperParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--checkpoint", default=-1, type=int)
    parser.add_argument("--save-images", action="store_true")
    args = parser.parse_args()
    print("args:", args)

    # params
    params_path = os.path.join(args.job_dir, "params.json")
    params = HyperParams.load(params_path)
    print("params:", params)

    # seed
    random.seed(params.seed)
    np.random.seed(params.seed)
    tf.random.set_seed(params.seed)

    # models
    env = create_env(params.env_name)
    model = Model(
        observation_space=env.observation_space,
        action_space=env.action_space,
        use_l2rl=params.use_l2rl,
    )

    # strategies
    policy = pyrl.strategies.Mode(model)

    # checkpoints
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, directory=args.job_dir, max_to_keep=None
    )
    checkpoint.restore(checkpoint_manager.checkpoints[args.checkpoint]).expect_partial()

    for env_name in constants.env_names:
        if args.env_name is not None and env_name != args.env_name:
            continue

        seed = params.eval_seed(env_name)

        transitions = collect_transitions(
            env_name=env_name,
            episodes=params.episodes_eval,
            policy=policy,
            seed=seed,
            params=params,
            render=True,
        )

        episodic_rewards = np.mean(np.sum(transitions["rewards"], axis=-1))
        print("episodic_rewards/eval/{}".format(env_name), episodic_rewards)

        if args.save_images:
            save_images(args.job_dir, env_name, transitions)


if __name__ == "__main__":
    main()
