import pyoneer as pynr
import pyoneer.rl as pyrl

from tool_use.env import create_env
from tool_use.model import Model
from tool_use.rollout import Rollout
from tool_use.parallel_rollout import ParallelRollout


def main():
    env_name = "TrapTube-v0"
    seed = 0
    max_episode_steps = 100
    episodes = 1024
    batch_size = 4

    # async vectorized rollout
    with pynr.debugging.Stopwatch() as stopwatch:
        env = pyrl.wrappers.BatchProcess(
            lambda: create_env(env_name, seed), batch_size=batch_size
        )
        model = Model(action_space=env.action_space)
        policy = pyrl.strategies.Sample(model)
        rollout = ParallelRollout(env, max_episode_steps)
        rollout(policy, episodes)
    print("vec:", stopwatch.duration)

    # vectorized rollout
    with pynr.debugging.Stopwatch() as stopwatch:
        env = pyrl.wrappers.Batch(
            lambda: create_env(env_name, seed), batch_size=batch_size
        )
        model = Model(action_space=env.action_space)
        policy = pyrl.strategies.Sample(model)
        rollout = ParallelRollout(env, max_episode_steps)
        rollout(policy, episodes)
    print("vec:", stopwatch.duration)

    # naive rollout
    with pynr.debugging.Stopwatch() as stopwatch:
        env = create_env(env_name, seed)
        model = Model(action_space=env.action_space)
        policy = pyrl.strategies.Sample(model)
        rollout = Rollout(env, max_episode_steps)
        rollout(policy, episodes)
    print("rollout:", stopwatch.duration)

    # ray rollout
    # with pynr.debugging.Stopwatch() as stopwatch:
    #     ray.get([rollout.remote() for i in range(4)])
    #     parallel_rollout(
    #         model, env_name, max_episode_steps, episodes, seed, training=None
    #     )
    # print("ray:", stopwatch.duration)


if __name__ == "__main__":
    main()
