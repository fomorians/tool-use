import numpy as np


class BatchRollout:
    def __init__(self, env):
        self.env = env

    def __call__(self, policy, episodes, render_mode=None):
        observation_space = self.env.observation_space
        action_space = self.env.action_space
        max_episode_steps = self.env.spec.max_episode_steps

        env_batch_size = len(self.env)
        batch_size = min(env_batch_size, episodes)
        batches = episodes // batch_size

        observations = np.zeros(
            shape=(episodes, max_episode_steps) + observation_space.shape,
            dtype=observation_space.dtype,
        )
        actions = np.zeros(
            shape=(episodes, max_episode_steps) + action_space.shape,
            dtype=action_space.dtype,
        )
        actions_prev = np.zeros(
            shape=(episodes, max_episode_steps) + action_space.shape,
            dtype=action_space.dtype,
        )
        observations_next = np.zeros(
            shape=(episodes, max_episode_steps) + observation_space.shape,
            dtype=observation_space.dtype,
        )
        rewards = np.zeros(shape=(episodes, max_episode_steps), dtype=np.float32)
        rewards_prev = np.zeros(shape=(episodes, max_episode_steps), dtype=np.float32)
        weights = np.zeros(shape=(episodes, max_episode_steps), dtype=np.float32)
        dones = np.ones(shape=(episodes, max_episode_steps), dtype=np.bool)

        if render_mode == "rgb_array":
            images = []

        for batch in range(batches):
            batch_start = batch * batch_size
            batch_end = batch_start + batch_size

            episode_done = np.zeros(shape=batch_size, dtype=np.bool)

            observation = self.env.reset()
            action_prev = np.zeros(
                shape=(env_batch_size,) + action_space.shape, dtype=action_space.dtype
            )
            reward_prev = np.zeros(shape=(env_batch_size,), dtype=np.float32)

            for step in range(max_episode_steps):
                if render_mode == "rgb_array":
                    images.append(self.env.render(mode="rgb_array"))
                elif render_mode is not None:
                    self.env.render(mode=render_mode)

                reset_state = step == 0

                observation = observation.astype(observation_space.dtype)
                inputs = {
                    "observations": observation[:, None, ...],
                    "actions_prev": action_prev[:, None, ...],
                }
                actions_batch = policy(inputs, training=False, reset_state=reset_state)
                action = actions_batch[:, 0].numpy()
                action = action.astype(action_space.dtype)

                observation_next, reward, done, info = self.env.step(action)

                observations[batch_start:batch_end, step] = observation[:batch_size]
                actions[batch_start:batch_end, step] = action[:batch_size]
                actions_prev[batch_start:batch_end, step] = action_prev[:batch_size]
                observations_next[batch_start:batch_end, step] = observation_next[
                    :batch_size
                ]
                rewards[batch_start:batch_end, step] = reward[:batch_size]
                rewards_prev[batch_start:batch_end, step] = reward_prev[:batch_size]
                weights[batch_start:batch_end, step] = np.where(
                    episode_done[:batch_size], 0.0, 1.0
                )
                dones[batch_start:batch_end, step] = done[:batch_size]

                episode_done = episode_done | done[:batch_size]

                # end the rollout if all episodes are done
                if np.all(episode_done):
                    break

                observation = observation_next
                action_prev = action
                reward_prev = np.asarray(reward, dtype=np.float32)

        transitions = {
            "observations": observations,
            "actions": actions,
            "actions_prev": actions_prev,
            "observations_next": observations_next,
            "rewards": rewards,
            "rewards_prev": rewards_prev,
            "weights": weights,
            "dones": dones,
        }

        if render_mode == "rgb_array":
            transitions["images"] = np.concatenate(images, axis=0)

        return transitions
