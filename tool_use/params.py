import attr


@attr.s
class HyperParams:
    learning_rate = attr.ib(default=1e-3)
    grad_clipping = attr.ib(default=10.0)
    batch_size = attr.ib(default=None)
    iters = attr.ib(default=100)
    epochs = attr.ib(default=10)
    episodes = attr.ib(default=10)
    max_episode_steps = attr.ib(default=500)
    decay = attr.ib(default=0.99)
    lambda_ = attr.ib(default=0.95)
