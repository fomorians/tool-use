import os
import random
import argparse
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from tool_use.params import HyperParams
from tool_use.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', required=True)
    parser.add_argument('--env', default='Pendulum-v0')
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()
    print(args)

    # make job directory
    os.makedirs(args.job_dir, exist_ok=True)

    # params
    params = HyperParams(env=args.env, seed=args.seed)
    params_path = os.path.join(args.job_dir, 'params.json')
    params.save(params_path)
    print(params)

    # eager
    tf.enable_eager_execution()

    # seed
    random.seed(params.seed)
    np.random.seed(params.seed)
    tf.set_random_seed(params.seed)

    # GPUs
    print('GPU Available:', tf.test.is_gpu_available())
    print('GPU Name:', tf.test.gpu_device_name())
    print('# of GPUs:', tfe.num_gpus())

    trainer = Trainer(args.job_dir, params)
    trainer.train()


if __name__ == '__main__':
    main()
