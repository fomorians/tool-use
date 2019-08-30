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
    train_iters = attr.ib(default=500, type=int)
    episodes_max = attr.ib(default=10240, type=int)
    episodes_train = attr.ib(default=128, type=int)
    episodes_eval = attr.ib(default=128, type=int)
    episodes_init = attr.ib(default=128, type=int)

    # optimization
    learning_rate = attr.ib(default=1e-3, type=float)
    grad_clipping = attr.ib(default=1.0, type=float)
    batch_size = attr.ib(default=16, type=int)
    l2_scale = attr.ib(default=1e-4, type=float)

    # sac
    num_critics = attr.ib(default=2, type=int)
    target_update_rate = attr.ib(default=1e-2, type=float)
    target_entropy = attr.ib(default=-4.0, type=float)

    @staticmethod
    def load(path):
        with tf.io.gfile.GFile(path, mode="r") as fp:
            data = json.load(fp)
        params = HyperParams(**data)
        return params

    def save(self, path):
        with tf.io.gfile.GFile(path, mode="w") as fp:
            json.dump(attr.asdict(self), fp)
