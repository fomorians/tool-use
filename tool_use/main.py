import attr
import trfl
import random
import argparse
import numpy as np
import tensorflow as tf
import pyoneer.rl as pyrl

from tqdm import trange

from tool_use.env import KukaEnv
from tool_use.models import Policy, Value
from tool_use.rollout import Rollout


@attr.s
class HyperParams:
    learning_rate = attr.ib(default=1e-3)
    iters = attr.ib(default=10)
    epochs = attr.ib(default=10)
    episodes = attr.ib(default=10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', required=True)
    parser.add_argument('--seed', default=42)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    tf.enable_eager_execution()

    env = KukaEnv(render=args.render)

    env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    summary_writer = tf.contrib.summary.create_file_writer(args.job_dir)
    summary_writer.set_as_default()

    action_size = env.action_space.shape[-1]
    params = HyperParams()
    rollout = Rollout(env, max_episode_steps=1000)
    policy = Policy(action_size=action_size)
    behavioral_policy = Policy(action_size=action_size)
    value = Value()
    strategy = pyrl.strategies.SampleStrategy(behavioral_policy)
    global_step = tf.train.create_global_step()
    optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)

    agent = pyrl.agents.ProximalPolicyOptimizationAgent(
        policy=policy,
        behavioral_policy=behavioral_policy,
        value=value,
        optimizer=optimizer)

    for it in trange(params.iters):
        states, actions, rewards, next_states, weights = rollout(
            strategy, episodes=params.episodes)
        dataset = tf.data.Dataset.from_tensors((states, actions, rewards,
                                                next_states, weights))

        for states, actions, rewards, next_states, weights in dataset:
            for epoch in range(params.epochs):
                agent.fit(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    weights=weights,
                    global_step=global_step)

                with tf.contrib.summary.always_record_summaries():
                    episodic_reward = tf.reduce_mean(
                        tf.reduce_sum(rewards, axis=-1))
                    tf.contrib.summary.scalar('policy_gradient_loss/train',
                                              agent.policy_gradient_loss)
                    tf.contrib.summary.scalar(
                        'policy_gradient_entropy_loss/train',
                        agent.policy_gradient_entropy_loss)
                    tf.contrib.summary.scalar('value_loss/train',
                                              agent.value_loss)
                    tf.contrib.summary.scalar('rewards/train', episodic_reward)

        trfl.update_target_variables(
            target_variables=behavioral_policy.trainable_variables,
            source_variables=policy.trainable_variables)


if __name__ == '__main__':
    main()
