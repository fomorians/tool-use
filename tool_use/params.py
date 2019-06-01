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
    train_iters = attr.ib(default=200)
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
    forward_coef = attr.ib(default=1.0)
    inverse_coef = attr.ib(default=1.0)
    intrinsic_coef = attr.ib(default=10.0)
    l2_coef = attr.ib(default=0.0)

    # optimization
    learning_rate = attr.ib(default=1e-3)
    grad_clipping = attr.ib(default=1.0)

    # PPO
    epsilon_clipping = attr.ib(default=0.2)
    discount_factor = attr.ib(default=0.99)
    lambda_factor = attr.ib(default=0.95)

    @property
    def eval_env_names(self):
        return [
            "PerceptualTrapTube-v0",
            "StructuralTrapTube-v0",
            "SymbolicTrapTube-v0",
            self.env_name,
        ]

    def train_seed(self, it):
        # ensure that no training iteration uses the same environment seed
        # care must be taken as the environments are batched and individually
        # seeded by the addition of the batch index
        # 0 + 4 * 128 + 0 * 128 = 512-639
        # 0 + 4 * 128 + 1 * 128 = 640–767
        # 0 + 4 * 128 + 2 * 128 = 768-895
        # 0 + 4 * 128 + 3 * 128 = 896-1023
        return (
            self.seed
            + len(self.eval_env_names) * self.env_batch_size
            + it * self.env_batch_size
        )

    def eval_seed(self, env_name):
        # ensure that each evaluation uses the same environment seed
        # care must be taken as the environments are batched and individually
        # seeded by the addition of the batch index
        # 0 + 0 * 128 = 0-127
        # 0 + 1 * 128 = 128-255
        # 0 + 2 * 128 = 256-383
        # 0 + 3 * 128 = 384-511
        return self.seed + self.eval_env_names.index(env_name) * self.env_batch_size

    @staticmethod
    def load(path):
        with tf.io.gfile.GFile(path, mode="r") as fp:
            data = json.load(fp)
        params = HyperParams(**data)
        return params

    def save(self, path):
        with tf.io.gfile.GFile(path, mode="w") as fp:
            json.dump(attr.asdict(self), fp)
