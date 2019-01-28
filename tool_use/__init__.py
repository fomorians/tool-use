import gym

# register kuka env
gym.envs.register(
    id='KukaEnv-v0',
    entry_point='tool_use.kuka_env:KukaEnv',
    max_episode_steps=200,
    kwargs=dict(should_render=False))
gym.envs.register(
    id='KukaEnvRender-v0',
    entry_point='tool_use.kuka_env:KukaEnv',
    max_episode_steps=200,
    kwargs=dict(should_render=True))
