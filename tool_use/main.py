import os
import random
import argparse
import numpy as np
import tensorflow as tf

from tool_use.params import HyperParams
from tool_use.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--l2rl", action="store_true")
    parser.add_argument("--intrinsic-reward", action="store_true")
    args = parser.parse_args()
    print(args)

    # make job directory
    os.makedirs(args.job_dir, exist_ok=True)

    # params
    params = HyperParams(env_name=args.env_name, seed=args.seed)
    params_path = os.path.join(args.job_dir, "params.json")
    params.save(params_path)
    print(params)

    # seed
    random.seed(params.seed)
    np.random.seed(params.seed)
    tf.random.set_seed(params.seed)

    # GPUs
    print("GPU Available:", tf.test.is_gpu_available())
    print("GPU Name:", tf.test.gpu_device_name())

    trainer = Trainer(args, params)
    trainer.train()


if __name__ == "__main__":
    main()
