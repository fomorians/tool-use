import numpy as np


class Rollout:
    def __init__(self, env, max_episode_steps):
        self.env = env
        self.max_episode_steps = max_episode_steps

    def __call__(self, policy, episodes, render=False):
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.shape[0]

        states = np.zeros(
            shape=(episodes, self.max_episode_steps, state_size),
            dtype=np.float32)
        actions = np.zeros(
            shape=(episodes, self.max_episode_steps, action_size),
            dtype=np.float32)
        next_states = np.zeros(
            shape=(episodes, self.max_episode_steps, state_size),
            dtype=np.float32)
        rewards = np.zeros(
            shape=(episodes, self.max_episode_steps), dtype=np.float32)
        weights = np.zeros(
            shape=(episodes, self.max_episode_steps), dtype=np.float32)

        for episode in range(episodes):
            state = self.env.reset()

            for step in range(self.max_episode_steps):
                if render:
                    self.env.render()

                reset_state = (step == 0)
                action = policy(state, reset_state=reset_state)
                next_state, reward, done, info = self.env.step(action)

                states[episode, step] = state
                actions[episode, step] = action
                next_states[episode, step] = next_state
                rewards[episode, step] = reward
                weights[episode, step] = 1.0

                if done:
                    break

                state = next_state

        return states, actions, rewards, next_states, weights
