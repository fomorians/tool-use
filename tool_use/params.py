import attr
import json


@attr.s
class HyperParams:
    # training
    iters = attr.ib(default=500)
    eval_interval = attr.ib(default=10)
    episodes = attr.ib(default=10)
    epochs = attr.ib(default=10)

    # losses
    value_coef = attr.ib(default=1.0)
    entropy_coef = attr.ib(default=0.01)

    # optimization
    learning_rate = attr.ib(default=1e-3)
    grad_clipping = attr.ib(default=10.0)

    # PPO
    epsilon_clipping = attr.ib(default=0.2)
    discount_factor = attr.ib(default=0.99)
    lambda_factor = attr.ib(default=0.95)
    scale = attr.ib(default=1.0)

    def save(self, path):
        with open(path, 'w') as fp:
            json.dump(attr.asdict(self), fp)
