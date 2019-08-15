import os
import time
import random
import imageio
import argparse
import numpy as np
import tensorflow as tf

import pyoneer.rl as pyrl

from tool_use.model import Model
from tool_use.params import HyperParams
from tool_use.env import create_env, collect_transitions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--checkpoint", default=-1, type=int)
    parser.add_argument("--env-name")
    parser.add_argument("--l2rl", action="store_true")
    args = parser.parse_args()
    print(args)

    # params
    params_path = os.path.join(args.job_dir, "params.json")
    params = HyperParams.load(params_path)
    save_dir = args.job_dir.replace("gs://tool-use-jobs", "jobs")
    print(params)

    # seed
    random.seed(params.seed)
    np.random.seed(params.seed)
    tf.random.set_seed(params.seed)

    # models
    env = create_env(params.env_name)
    model = Model(
        observation_space=env.observation_space,
        action_space=env.action_space,
        l2rl=args.l2rl,
    )

    # strategies
    policy = pyrl.strategies.Mode(model)

    # checkpoints
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, directory=args.job_dir, max_to_keep=None
    )
    checkpoint.restore(checkpoint_manager.checkpoints[args.checkpoint]).expect_partial()

    for env_name in params.eval_env_names:
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
            return_hidden_states=True,
        )

        path = os.path.join(save_dir, "transition_data", env_name)
        os.makedirs(path, exist_ok=True)

        for i in range(params.episodes_eval):
            transition = {key: transitions[key][i] for key in transitions}
            file = os.path.join(path, f"{i}.npz")
            np.savez(file, **transition)

        episodic_rewards = np.mean(np.sum(transitions["rewards"], axis=-1))
        print("episodic_rewards/eval/{}".format(env_name), episodic_rewards)


if __name__ == "__main__":
    main()
