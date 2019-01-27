import numpy as np
import tensorflow as tf


# TODO: Use concat approach to work with regular rollout. Multiprocessing?
class ParallelRollout:
    def __init__(self, env, max_episode_steps):
        self.env = env
        self.max_episode_steps = max_episode_steps

    def __call__(self, policy, episodes):
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.shape[0]

        batch_size = len(self.env)
        batches = int(np.ceil(episodes / batch_size))

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

        for batch in range(batches):
            batch_start = batch * batch_size
            batch_end = batch_start + batch_size

            diff = batch_end - episodes
            slice_size = batch_size

            if diff > 0:
                slice_size = batch_size - diff
                batch_end = batch_start + slice_size

            episode_done = np.zeros(shape=slice_size, dtype=np.bool)

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

                states[batch_start:batch_end, step] = state[:slice_size]
                actions[batch_start:batch_end, step] = action[:slice_size]
                next_states[batch_start:batch_end, step] = (
                    next_state[:slice_size])
                rewards[batch_start:batch_end, step] = reward[:slice_size]

                for i in range(slice_size):
                    # if the ith rollout is not done set the weight to 1
                    if not episode_done[i]:
                        weights[batch_start + i, step] = 1.0

                    # if the ith rollout is done mark it as done
                    episode_done[i] = episode_done[i] or done[i]

                # end the rollout if all episodes are done
                if np.all(episode_done):
                    break

                state = next_state

        return states, actions, rewards, next_states, weights
