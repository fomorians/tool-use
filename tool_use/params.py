import attr
import json
import tensorflow as tf


@attr.s
class HyperParams:
    # environment
    env_name = attr.ib(default=None)
    seed = attr.ib(default=42)
    max_episode_steps = attr.ib(default=50)

    # training
    train_iters = attr.ib(default=10)
    episodes_train = attr.ib(default=1024)
    episodes_eval = attr.ib(default=128)
    epochs = attr.ib(default=10)
    batch_size = attr.ib(default=128)
    reward_decay = attr.ib(default=0.9)
    center_reward = attr.ib(default=False)
    normalize_advantages = attr.ib(default=True)

    # losses
    value_coef = attr.ib(default=1e-1)
    entropy_coef = attr.ib(default=0.0)
    l2_coef = attr.ib(default=0.0)

    # optimization
    learning_rate = attr.ib(default=1e-3)
    grad_clipping = attr.ib(default=None)

    # PPO
    epsilon_clipping = attr.ib(default=0.2)
    discount_factor = attr.ib(default=0.99)
    lambda_factor = attr.ib(default=0.95)

    def save(self, path):
        with tf.io.gfile.GFile(path, mode="w") as fp:
            json.dump(attr.asdict(self), fp)
