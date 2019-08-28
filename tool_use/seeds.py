class Seeds:
    """
    Simple seed management. Ensures that no training iteration uses the
    same environment seed and that each evaluation uses the same
    environment seed.

    Care must be taken as the environments are batched and individually
    seeded by the addition of the batch index.
    """

    def __init__(self, base_seed, env_batch_size, eval_env_names):
        self.base_seed = base_seed
        self.env_batch_size = env_batch_size
        self.eval_env_names = eval_env_names

    def train_seed(self, it):
        # 0 + 4 * 128 + 0 * 128 = 512-639
        # 0 + 4 * 128 + 1 * 128 = 640–767
        # 0 + 4 * 128 + 2 * 128 = 768-895
        # 0 + 4 * 128 + 3 * 128 = 896-1023
        return (
            self.base_seed
            + len(self.eval_env_names) * self.env_batch_size
            + it * self.env_batch_size
        )

    def eval_seed(self, env_name):
        # 0 + 0 * 128 = 0-127
        # 0 + 1 * 128 = 128-255
        # 0 + 2 * 128 = 256-383
        # 0 + 3 * 128 = 384-511
        return self.base_seed + self.eval_env_names.index(env_name) * self.env_batch_size
