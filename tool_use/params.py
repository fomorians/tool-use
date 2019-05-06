import attr
import json
import tensorflow as tf


@attr.s
class HyperParams:
    # environment
    seed = attr.ib(default=42)
    max_episode_steps = attr.ib(default=100)

    # training
    train_iters = attr.ib(default=30)
    episodes = attr.ib(default=128)
    epochs = attr.ib(default=10)
    batch_size = attr.ib(default=32)
    eval_interval = attr.ib(default=10)
    reward_decay = attr.ib(default=0.9)
    center_reward = attr.ib(default=False)
    normalize_advantages = attr.ib(default=True)

    # losses
    value_coef = attr.ib(default=1e-3)
    entropy_coef = attr.ib(default=0.02)
    l2_coef = attr.ib(default=0.0)

    # optimization
    learning_rate = attr.ib(default=1e-3)
    grad_clipping = attr.ib(default=1.0)

    # PPO
    epsilon_clipping = attr.ib(default=0.2)
    discount_factor = attr.ib(default=0.99)
    lambda_factor = attr.ib(default=0.95)

    def save(self, path):
        with tf.io.gfile.GFile(path, mode="w") as fp:
            json.dump(attr.asdict(self), fp)
