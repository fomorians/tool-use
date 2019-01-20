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
        kwargs=dict(should_render=True))

    env = gym.make(args.env)

    for episode in range(args.episodes):
        state = env.reset()

        for step in range(env.spec.max_episode_steps):
            if args.env != 'KukaEnv-v0':
                env.render()

            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            print(reward)

            if done:
                break

            state = next_state


if __name__ == '__main__':
    main()
