import pyoneer as pynr
import pyoneer.rl as pyrl

from tool_use.env import create_env
from tool_use.model import Model
from tool_use.batch_rollout import BatchRollout


def main():
    env_name = "TrapTube-v0"
    seed = 0
    max_episode_steps = 100
    episodes = 1024
    batch_size = 128

    # # async vectorized rollout
    # with pynr.debugging.Stopwatch() as stopwatch:
    #     env = pyrl.wrappers.BatchProcess(
    #         lambda: create_env(env_name, seed), batch_size=batch_size
    #     )
    #     model = Model(action_space=env.action_space)
    #     policy = pyrl.strategies.Sample(model)
    #     rollout = BatchRollout(env, max_episode_steps)
    #     rollout(policy, episodes)
    # print("batch process:", stopwatch.duration)
    # env.close()

    # vectorized rollout
    with pynr.debugging.Stopwatch() as stopwatch:
        env = pyrl.wrappers.Batch(
            lambda: create_env(env_name, seed), batch_size=batch_size
        )
        model = Model(action_space=env.action_space)
        policy = pyrl.strategies.Sample(model)
        rollout = BatchRollout(env, max_episode_steps)
        rollout(policy, episodes)
    print("batch:", stopwatch.duration)
    env.close()

    # # naive rollout
    # with pynr.debugging.Stopwatch() as stopwatch:
    #     env = create_env(env_name, seed)
    #     model = Model(action_space=env.action_space)
    #     policy = pyrl.strategies.Sample(model)
    #     rollout = Rollout(env, max_episode_steps)
    #     rollout(policy, episodes)
    # print("rollout:", stopwatch.duration)
    # env.close()

    # ray rollout
    # ray.init()
    # with pynr.debugging.Stopwatch() as stopwatch:
    #     ray.get([ray_rollout.remote() for i in range(batch_size)])
    # print("ray:", stopwatch.duration)


if __name__ == "__main__":
    main()
