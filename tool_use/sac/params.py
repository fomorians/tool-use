import attr
import json
import tensorflow as tf


@attr.s
class HyperParams:
    # environment
    env_name = attr.ib(default=None, type=str)
    env_batch_size = attr.ib(default=128, type=int)
    seed = attr.ib(default=None, type=int)
    discount_factor = attr.ib(default=0.99, type=float)

    # training
    train_iters = attr.ib(default=1000, type=int)
    episodes_train = attr.ib(default=8, type=int)
    episodes_eval = attr.ib(default=128, type=int)

    # optimization
    learning_rate = attr.ib(default=1e-3, type=float)
    grad_clipping = attr.ib(default=1.0, type=float)
    batch_size = attr.ib(default=128, type=int)

    # sac
    target_update_rate = attr.ib(default=1e-2, type=float)
    target_entropy = attr.ib(default=-1.0, type=float)
    num_q_fns = attr.ib(default=5, type=int)
    max_size = attr.ib(default=int(1e6), type=int)
    num_samples = attr.ib(default=1024, type=int)
    steps_init = attr.ib(default=int(1e3), type=int)

    @staticmethod
    def load(path):
        with tf.io.gfile.GFile(path, mode="r") as fp:
            data = json.load(fp)
        params = HyperParams(**data)
        return params

    def save(self, path):
        with tf.io.gfile.GFile(path, mode="w") as fp:
            json.dump(attr.asdict(self), fp)
