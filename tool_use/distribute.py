import os
import random
import numpy as np
import tensorflow as tf
import multiprocessing
import pyoneer as pynr
import pyoneer.rl as pyrl

from tool_use.env import create_env
from tool_use.model import Model
from tool_use.rollout import Rollout


def worker_fn(
    worker_id, model_weights, env_name, max_episode_steps, episodes, seed, training=None
):
    with pynr.debugging.Stopwatch() as stopwatch:
        # seed
        random.seed(seed + worker_id)
        np.random.seed(seed + worker_id)
        tf.random.set_seed(seed + worker_id)

        # environment
        env = create_env(env_name, seed + worker_id)

        # models
        local_model = Model(action_space=env.action_space)
        # local_model.set_weights(model_weights)

        # policies
        if training:
            policy = pyrl.strategies.Sample(local_model)
        else:
            policy = pyrl.strategies.Mode(local_model)

        # rollout
        rollout = Rollout(env, max_episode_steps)

        transitions = rollout(policy, episodes)
    print("worker_fn:", stopwatch.duration)
    return transitions


def parallel_rollout(model, env_name, max_episode_steps, episodes, seed, training=None):
    cpu_count = os.cpu_count()
    assert episodes % cpu_count == 0

    ctx = multiprocessing.get_context("spawn")
    pool = ctx.Pool(processes=cpu_count)

    worker_episodes = episodes // cpu_count
    model_weights = model.get_weights()

    iterable = [
        (
            worker_id,
            model_weights,
            env_name,
            max_episode_steps,
            worker_episodes,
            seed,
            training,
        )
        for worker_id in range(cpu_count)
    ]

    # aggregate transitions from each worker
    # transitions_list = []
    for worker_transitions in pool.starmap(worker_fn, iterable):
        # transitions_list.append(worker_transitions)
        pass

    # with pynr.debugging.Stopwatch() as stopwatch:
    # for key, val in worker_transitions.items():
    #     transitions[key].append(val)
    #     # concat along the episode dimension
    #     for key, val in transitions.items():
    #         transitions[key] = np.concatenate(transitions[key], axis=0)
    # print("parallel_rollout:", stopwatch.duration)
    # return transitions
