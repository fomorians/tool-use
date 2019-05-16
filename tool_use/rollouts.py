import numpy as np


class Rollout:
    def __init__(self, env, max_episode_steps):
        self.env = env
        self.max_episode_steps = max_episode_steps

    def __call__(self, policy, episodes, render=False, render_mode="rgb_array"):
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
        actions_prev = np.zeros(
            shape=(episodes, self.max_episode_steps) + action_space.shape,
            dtype=action_space.dtype,
        )
        observations_next = np.zeros(
            shape=(episodes, self.max_episode_steps) + observation_space.shape,
            dtype=action_space.dtype,
        )
        rewards = np.zeros(shape=(episodes, self.max_episode_steps), dtype=np.float32)
        rewards_prev = np.zeros(
            shape=(episodes, self.max_episode_steps), dtype=np.float32
        )
        weights = np.zeros(shape=(episodes, self.max_episode_steps), dtype=np.float32)

        images = None
        if render and render_mode == "rgb_array":
            images = []

        for episode in range(episodes):
            observation = self.env.reset()
            action_prev = np.zeros(shape=action_space.shape, dtype=action_space.dtype)
            reward_prev = np.zeros(shape=(), dtype=np.float32)

            if render and render_mode == "rgb_array":
                episode_images = []

            for step in range(self.max_episode_steps):
                if render:
                    if render_mode == "rgb_array":
                        image = self.env.render(mode=render_mode)
                        episode_images.append(image)
                    else:
                        self.env.render()

                reset_state = step == 0

                observation = observation.astype(observation_space.dtype)
                action_batch = policy(
                    observation[None, None, ...],
                    action_prev[None, None, ...],
                    reward_prev[None, None, ...],
                    training=False,
                    reset_state=reset_state,
                )
                action = action_batch[0, 0].numpy()
                action = action.astype(action_space.dtype)

                observation_next, reward, done, info = self.env.step(action)

                observations[episode, step] = observation
                actions[episode, step] = action
                actions_prev[episode, step] = action_prev
                observations_next[episode, step] = observation_next
                rewards[episode, step] = reward
                rewards_prev[episode, step] = reward_prev
                weights[episode, step] = 1.0

                if done:
                    if render:
                        if render_mode == "rgb_array":
                            image = self.env.render(mode=render_mode)
                            episode_images.append(image)
                            images.append(episode_images)
                        else:
                            self.env.render()
                    break

                observation = observation_next
                action_prev = action
                reward_prev = reward.astype(np.float32)

        return {
            "observations": observations,
            "actions": actions,
            "actions_prev": actions_prev,
            "observations_next": observations_next,
            "rewards": rewards,
            "rewards_prev": rewards_prev,
            "weights": weights,
            "images": images,
        }
