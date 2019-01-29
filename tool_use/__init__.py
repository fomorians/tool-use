import gym

max_episode_steps = 200

# register kuka env
gym.envs.register(
    id='KukaVelocityEnv-v0',
    entry_point='tool_use.kuka_env:KukaEnv',
    max_episode_steps=max_episode_steps,
    kwargs=dict(should_render=False, position_control=False))
gym.envs.register(
    id='KukaVelocityRenderEnv-v0',
    entry_point='tool_use.kuka_env:KukaEnv',
    max_episode_steps=max_episode_steps,
    kwargs=dict(should_render=True, position_control=False))
gym.envs.register(
    id='KukaPositionEnv-v0',
    entry_point='tool_use.kuka_env:KukaEnv',
    max_episode_steps=max_episode_steps,
    kwargs=dict(should_render=False, position_control=True))
gym.envs.register(
    id='KukaPositionRenderEnv-v0',
    entry_point='tool_use.kuka_env:KukaEnv',
    max_episode_steps=max_episode_steps,
    kwargs=dict(should_render=True, position_control=True))
