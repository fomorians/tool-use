import numpy as np
import tensorflow as tf


class ParallelRollout:
    def __init__(self, env, max_episode_steps):
        self.env = env
        self.max_episode_steps = max_episode_steps

    def __call__(self, policy):
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.shape[0]

        episodes = len(self.env)

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
        episode_done = np.zeros(shape=episodes, dtype=np.bool)

        state = self.env.reset()

        for step in range(self.max_episode_steps):
            reset_state = (step == 0)

            state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
            action_batch = policy(
                state_tensor[:, None, ...],
                training=False,
                reset_state=reset_state)
            action = action_batch[:, 0].numpy()

            next_state, reward, done, info = self.env.step(action)

            states[:, step] = state
            actions[:, step] = action
            next_states[:, step] = next_state
            rewards[:, step] = reward

            for episode in range(episodes):
                # if the episode is not done set the weight to 1
                if not episode_done[episode]:
                    weights[episode, step] = 1.0

                # if the episode is done mark it as done
                episode_done[episode] = episode_done[episode] or done[episode]

            # end the rollout if all episodes are done
            if np.all(episode_done):
                break

            state = next_state

        return states, actions, rewards, next_states, weights
