import gym
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='Pendulum-v0')
    parser.add_argument('--seed', default=42)
    parser.add_argument('--episodes', default=10)
    args = parser.parse_args()

    gym.envs.register(
        id='KukaEnv-v0',
        entry_point='tool_use.kuka_env:KukaEnv',
        max_episode_steps=1000,
        kwargs=dict(render=True))

    env = gym.make(args.env)
    for _ in range(args.episodes):
        env.reset()
        for _ in range(env.spec.max_episode_steps):
            if args.env != 'KukaEnv-v0':
                env.render()

            action = env.action_space.sample()
            _, reward, _, _ = env.step(action)
            print(reward)


if __name__ == '__main__':
    main()
