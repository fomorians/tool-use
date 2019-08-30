import os
import random
import argparse
import numpy as np
import tensorflow as tf

from tool_use import constants
from tool_use.sac.params import HyperParams
from tool_use.sac.algorithm import Algorithm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--env", required=True)
    args = parser.parse_args()
    print("args:", args)

    # make job directory
    os.makedirs(args.job_dir, exist_ok=True)

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # hyperparams
    params = HyperParams(env_name=args.env, seed=args.seed)
    params_path = os.path.join(args.job_dir, "params.json")
    params.save(params_path)
    print("params:", params)

    # GPUs
    print("GPU Available:", tf.test.is_gpu_available())
    print("GPU Name:", tf.test.gpu_device_name())

    # training
    algorithm = Algorithm(args.job_dir, params, constants.env_names)
    algorithm.train()


if __name__ == "__main__":
    main()
