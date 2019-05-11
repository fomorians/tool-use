import numpy as np
import tensorflow as tf


class Rollout:
    def __init__(self, env, max_episode_steps):
        self.env = env
        self.max_episode_steps = max_episode_steps

    def __call__(self, policy, episodes, render=False):
        observation_space = self.env.observation_space
        action_space = self.env.action_space

        observations = np.zeros(
            shape=(episodes, self.max_episode_steps) + observation_space.shape,
            dtype=observation_space.dtype,
        )
        actions = np.zeros(
            shape=(episodes, self.max_episode_steps) + action_space.shape,
            dtype=action_space.dtype,
        )
        observations_next = np.zeros(
            shape=(episodes, self.max_episode_steps) + observation_space.shape,
            dtype=action_space.dtype,
        )
        rewards = np.zeros(shape=(episodes, self.max_episode_steps), dtype=np.float32)
        weights = np.zeros(shape=(episodes, self.max_episode_steps), dtype=np.float32)

        for episode in range(episodes):
            observation = self.env.reset()

            for step in range(self.max_episode_steps):
                if render:
                    self.env.render()

                reset_state = step == 0

                observation_tensor = tf.convert_to_tensor(
                    observation, dtype=observation_space.dtype
                )
                action_batch = policy(
                    observation_tensor[None, None, ...],
                    training=False,
                    reset_state=reset_state,
                )
                action = action_batch[0, 0].numpy()

                observation_next, reward, done, info = self.env.step(action)

                observations[episode, step] = observation
                actions[episode, step] = action
                observations_next[episode, step] = observation_next
                rewards[episode, step] = reward
                weights[episode, step] = 1.0

                if done:
                    break

                observation = observation_next

        # ensure rewards are masked
        rewards *= weights

        return observations, actions, rewards, observations_next, weights
