class FnStrategy:
    """
    Specifies the action generation function of the agent to use.
    """

    def __init__(self, agent, fn_name):
        self.agent = agent
        self.fn_name = fn_name

    def __call__(self, *args, **kwargs):
        strategy = getattr(self.agent, self.fn_name)
        actions = strategy(*args, **kwargs)
        return actions
