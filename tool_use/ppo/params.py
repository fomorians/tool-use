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
    episodes_train = attr.ib(default=1024, type=int)
    episodes_eval = attr.ib(default=128, type=int)

    # losses
    value_coef = attr.ib(default=1e-2, type=float)
    entropy_coef = attr.ib(default=1e-4, type=float)
    forward_coef = attr.ib(default=1.0, type=float)
    inverse_coef = attr.ib(default=1.0, type=float)
    intrinsic_coef = attr.ib(default=10.0, type=float)
    l2_coef = attr.ib(default=0.0, type=float)

    # optimization
    learning_rate = attr.ib(default=1e-3, type=float)
    grad_clipping = attr.ib(default=1.0, type=float)
    batch_size = attr.ib(default=128, type=int)

    # PPO
    epsilon_clipping = attr.ib(default=0.2, type=float)
    lambda_factor = attr.ib(default=0.95, type=float)
    epochs = attr.ib(default=10, type=int)
    reward_decay = attr.ib(default=0.9, type=float)
    center_reward = attr.ib(default=False, type=bool)
    normalize_advantages = attr.ib(default=True, type=bool)
    intrinsic_scale = attr.ib(default=0.1, type=float)
    use_l2rl = attr.ib(default=False, type=bool)
    use_icm = attr.ib(default=True, type=bool)

    @staticmethod
    def load(path):
        with tf.io.gfile.GFile(path, mode="r") as fp:
            data = json.load(fp)
        params = HyperParams(**data)
        return params

    def save(self, path):
        with tf.io.gfile.GFile(path, mode="w") as fp:
            json.dump(attr.asdict(self), fp)
