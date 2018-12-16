import numpy as np
import multiprocessing

from tool_use.process_env import ProcessEnv


class BatchEnv(object):
    """
    Adapted from Batch PPO:
    https://github.com/google-research/batch-ppo/blob/master/agents/tools/batch_env.py
    """

    def __init__(self, constructor, size=None, blocking=True):
        if size is None:
            size = multiprocessing.cpu_count()

        self.envs = [ProcessEnv(constructor) for env in range(size)]
        self.blocking = blocking

        observation_space = self.envs[0].observation_space
        if not all(env.observation_space == observation_space
                   for env in self.envs):
            raise ValueError(
                'All environments must use the same observation space.')

        action_space = self.envs[0].action_space
        if not all(env.action_space == action_space for env in self.envs):
            raise ValueError(
                'All environments must use the same observation space.')

    def __len__(self):
        return len(self.envs)

    def __getitem__(self, index):
        return self.envs[index]

    def __getattr__(self, name):
        return getattr(self.envs[0], name)

    def seed(self, seed=None):
        for i, env in enumerate(self.envs):
            env.seed(seed + i, blocking=True)

    def reset(self):
        if self.blocking:
            states = [env.reset() for env in self.envs]
        else:
            promises = [env.reset(blocking=False) for env in self.envs]
            states = [promise() for promise in promises]

        state = np.stack(states, axis=0)
        return state

    def step(self, actions):
        if self.blocking:
            transitions = [
                env.step(action, blocking=True)
                for env, action in zip(self.envs, actions)
            ]
        else:
            promises = [
                env.step(action, blocking=False)
                for env, action in zip(self.envs, actions)
            ]
            transitions = [promise() for promise in promises]

        states, rewards, dones, infos = zip(*transitions)
        state = np.stack(states, axis=0)
        reward = np.stack(rewards, axis=0)
        done = np.stack(dones, axis=0)
        info = tuple(infos)
        return state, reward, done, info

    def close(self):
        for env in self.envs:
            if hasattr(env, 'close'):
                env.close()
