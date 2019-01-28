import attr
import json


@attr.s
class HyperParams:
    # environment
    env = attr.ib()
    seed = attr.ib(default=42)

    # training
    train_iters = attr.ib(default=300)
    episodes = attr.ib(default=16)
    epochs = attr.ib(default=10)
    horizon = attr.ib(default=200)
    batch_size = attr.ib(default=16)
    eval_interval = attr.ib(default=10)
    reward_decay = attr.ib(default=0.9)

    # losses
    value_coef = attr.ib(default=1e-4)
    entropy_coef = attr.ib(default=0.0)

    # optimization
    learning_rate = attr.ib(default=3e-4)
    grad_clipping = attr.ib(default=10.0)

    # PPO
    epsilon_clipping = attr.ib(default=0.2)
    discount_factor = attr.ib(default=0.99)
    lambda_factor = attr.ib(default=0.95)
    scale = attr.ib(default=1.0)

    def save(self, path):
        with open(path, 'w') as fp:
            json.dump(attr.asdict(self), fp)
