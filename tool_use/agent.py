class Agent:
    """
    Interface between the policy network and the environment.
    """

    def __init__(self, policy):
        self.policy = policy

    def explore(self, *args, **kwargs):
        return self.policy.sample(*args, **kwargs)

    def exploit(self, *args, **kwargs):
        return self.policy.mode(*args, **kwargs)
