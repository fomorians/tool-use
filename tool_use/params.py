import attr


@attr.s
class HyperParams:
    learning_rate = attr.ib(default=1e-3)
    grad_clipping = attr.ib(default=10.0)
    batch_size = attr.ib(default=None)
    iters = attr.ib(default=30)
    epochs = attr.ib(default=10)
    episodes = attr.ib(default=10)
    max_episode_steps = attr.ib(default=200)
    discount_factor = attr.ib(default=0.995)
    lambda_factor = attr.ib(default=0.95)
    value_coef = attr.ib(default=1.0)
    scale = attr.ib(default=1.0)
    entropy_coef = attr.ib(default=0.05)
    epsilon_clipping = attr.ib(default=0.2)
