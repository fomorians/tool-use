import attr
import json
import tensorflow as tf


@attr.s
class HyperParams:
    # environment
    env = attr.ib()
    seed = attr.ib(default=42)

    # training
    train_iters = attr.ib(default=300)
    episodes = attr.ib(default=64)
    epochs = attr.ib(default=10)
    horizon = attr.ib(default=200)
    batch_size = attr.ib(default=64)
    eval_interval = attr.ib(default=10)
    reward_decay = attr.ib(default=0.9)
    center_reward = attr.ib(default=True)

    # losses
    value_coef = attr.ib(default=1e-4)
    entropy_coef = attr.ib(default=0.0)
    forward_coef = attr.ib(default=0.01)
    inverse_coef = attr.ib(default=0.01)
    l2_coef = attr.ib(default=0.0)

    # optimization
    learning_rate = attr.ib(default=1e-3)
    grad_clipping = attr.ib(default=1.0)

    # PPO
    epsilon_clipping = attr.ib(default=0.2)
    discount_factor = attr.ib(default=0.99)
    lambda_factor = attr.ib(default=0.95)
    scale = attr.ib(default=1.0)

    def save(self, path):
        with tf.gfile.GFile(path, mode='w') as fp:
            json.dump(attr.asdict(self), fp)
