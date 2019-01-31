import gym

max_episode_steps = 200

# register kuka env
gym.envs.register(
    id='KukaPositionEnv-v0',
    entry_point='tool_use.kuka_env:KukaEnv',
    max_episode_steps=max_episode_steps,
    kwargs=dict(should_render=False))
gym.envs.register(
    id='KukaPositionRenderEnv-v0',
    entry_point='tool_use.kuka_env:KukaEnv',
    max_episode_steps=max_episode_steps,
    kwargs=dict(
        should_render=True,
        enable_wind=False,
        enable_blocks=False,
        velocity_penalty=0.0))
