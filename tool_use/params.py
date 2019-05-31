import attr
import json
import tensorflow as tf


@attr.s
class HyperParams:
    # environment
    env_name = attr.ib(default=None)
    env_batch_size = attr.ib(default=128)
    seed = attr.ib(default=None)
    max_episode_steps = attr.ib(default=50)

    # training
    train_iters = attr.ib(default=50)
    episodes_train = attr.ib(default=1024)
    episodes_eval = attr.ib(default=128)
    epochs = attr.ib(default=10)
    batch_size = attr.ib(default=128)
    reward_decay = attr.ib(default=0.9)
    intrinsic_scale = attr.ib(default=0.1)
    center_reward = attr.ib(default=False)
    normalize_advantages = attr.ib(default=True)

    # losses
    value_coef = attr.ib(default=1e-2)
    entropy_coef = attr.ib(default=1e-4)
    forward_coef = attr.ib(default=0.2)
    inverse_coef = attr.ib(default=0.8)
    intrinsic_coef = attr.ib(default=10.0)
    l2_coef = attr.ib(default=0.0)

    # optimization
    learning_rate = attr.ib(default=1e-3)
    grad_clipping = attr.ib(default=1.0)

    # PPO
    epsilon_clipping = attr.ib(default=0.2)
    discount_factor = attr.ib(default=0.99)
    lambda_factor = attr.ib(default=0.95)

    @staticmethod
    def load(path):
        with tf.io.gfile.GFile(path, mode="r") as fp:
            data = json.load(fp)
        params = HyperParams(**data)
        return params

    def save(self, path):
        with tf.io.gfile.GFile(path, mode="w") as fp:
            json.dump(attr.asdict(self), fp)
