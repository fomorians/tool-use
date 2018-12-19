import attr


@attr.s
class HyperParams:
    # training
    iters = attr.ib(default=100)
    eval_interval = attr.ib(default=10)
    episodes = attr.ib(default=10)
    epochs = attr.ib(default=10)
    max_episode_steps = attr.ib(default=200)

    # losses
    value_coef = attr.ib(default=1e-3)
    entropy_coef = attr.ib(default=0.05)

    # optimization
    learning_rate = attr.ib(default=1e-3)
    grad_clipping = attr.ib(default=10.0)

    # PPO
    epsilon_clipping = attr.ib(default=0.2)
    discount_factor = attr.ib(default=0.99)
    lambda_factor = attr.ib(default=0.95)
    scale = attr.ib(default=1.0)
