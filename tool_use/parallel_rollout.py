import numpy as np


# TODO: replace with wrapper over a regular rollout that concats
class ParallelRollout:
    def __init__(self, env, max_episode_steps):
        self.env = env
        self.max_episode_steps = max_episode_steps

    def __call__(self, policy, episodes):
        observation_space = self.env.observation_space
        action_space = self.env.action_space

        batch_size = len(self.env)
        batches = int(np.ceil(episodes / batch_size))

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
            dtype=observation_space.dtype,
        )
        rewards = np.zeros(shape=(episodes, self.max_episode_steps), dtype=np.float32)
        rewards_prev = np.zeros(
            shape=(episodes, self.max_episode_steps), dtype=np.float32
        )
        weights = np.zeros(shape=(episodes, self.max_episode_steps), dtype=np.float32)

        for batch in range(batches):
            batch_start = batch * batch_size
            batch_end = batch_start + batch_size

            diff = batch_end - episodes
            slice_size = batch_size

            if diff > 0:
                slice_size = batch_size - diff
                batch_end = batch_start + slice_size

            episode_done = np.zeros(shape=slice_size, dtype=np.bool)

            observation = self.env.reset()
            action_prev = np.zeros(
                shape=(batch_size,) + action_space.shape, dtype=action_space.dtype
            )
            reward_prev = np.zeros(shape=(batch_size), dtype=np.float32)

            for step in range(self.max_episode_steps):
                reset_state = step == 0

                observation = observation.astype(observation_space.dtype)
                action_batch = policy(
                    observation[:, None, ...],
                    action_prev[:, None, ...],
                    reward_prev[:, None, ...],
                    training=False,
                    reset_state=reset_state,
                )
                action = action_batch[:, 0].numpy()
                action = action.astype(action_space.dtype)

                observation_next, reward, done, info = self.env.step(action)

                observations[batch_start:batch_end, step] = observation[:slice_size]
                actions[batch_start:batch_end, step] = action[:slice_size]
                actions_prev[batch_start:batch_end, step] = action_prev[:slice_size]
                observations_next[batch_start:batch_end, step] = observation_next[
                    :slice_size
                ]
                rewards[batch_start:batch_end, step] = reward[:slice_size]
                rewards_prev[batch_start:batch_end, step] = reward_prev[:slice_size]

                for i in range(slice_size):
                    # if the ith rollout is not done set the weight to 1
                    if not episode_done[i]:
                        weights[batch_start + i, step] = 1.0

                    # if the ith rollout is done mark it as done
                    episode_done[i] = episode_done[i] or done[i]

                # end the rollout if all episodes are done
                if np.all(episode_done):
                    break

                observation = observation_next
                action_prev = action
                reward_prev = reward.astype(np.float32)

        # ensure rewards are masked
        rewards *= weights

        return {
            "observations": observations,
            "actions": actions,
            "actions_prev": actions_prev,
            "observations_next": observations_next,
            "rewards": rewards,
            "rewards_prev": rewards_prev,
            "weights": weights,
        }
