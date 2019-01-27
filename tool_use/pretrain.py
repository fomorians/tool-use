import os
import gym
import attr
import json
import random
import argparse
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from tool_use import models
from tool_use.rollout import Rollout
from tool_use.strategy import RandomStrategy
from tool_use.wrappers import RangeNormalize


@attr.s
class SupervisedHyperParams:
    env = attr.ib()
    seed = attr.ib(default=42)
    epochs = attr.ib(default=100)
    episodes = attr.ib(default=1000)
    eval_episodes = attr.ib(default=100)
    batch_size = attr.ib(default=10)
    learning_rate = attr.ib(default=1e-3)
    grad_clipping = attr.ib(default=10)

    def save(self, path):
        with open(path, 'w') as fp:
            json.dump(attr.asdict(self), fp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', required=True)
    parser.add_argument('--env', default='Pendulum-v0')
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()
    print(args)

    # register kuka env
    gym.envs.register(
        id='KukaEnv-v0',
        entry_point='tool_use.kuka_env:KukaEnv',
        max_episode_steps=200,
        kwargs=dict(should_render=False))

    # make job directory
    if not os.path.exists(args.job_dir):
        os.makedirs(args.job_dir)

    # params
    params = SupervisedHyperParams(env=args.env, seed=args.seed)
    params_path = os.path.join(args.job_dir, 'params.json')
    params.save(params_path)
    print(params)

    # eager
    tf.enable_eager_execution()

    # GPUs
    print('GPU Available:', tf.test.is_gpu_available())
    print('GPU Name:', tf.test.gpu_device_name())
    print('# of GPUs:', tfe.num_gpus())

    # environment
    env = gym.make(params.env)
    env = RangeNormalize(env)

    # seeding
    env.seed(params.seed)
    random.seed(params.seed)
    np.random.seed(params.seed)
    tf.set_random_seed(params.seed)

    # optimization
    global_step = tf.train.create_global_step()
    optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)

    # models
    inverse_model = models.InverseModel(action_size=env.action_space.shape[0])
    forward_model = models.ForwardModel()
    state_embedding = models.StateEmbedding()
    action_embedding = models.ActionEmbedding()

    # checkpoints
    checkpoint = tf.train.Checkpoint(
        global_step=global_step,
        optimizer=optimizer,
        inverse_model=inverse_model,
        forward_model=forward_model,
        state_embedding=state_embedding,
        action_embedding=action_embedding)
    checkpoint_path = tf.train.latest_checkpoint(args.job_dir)
    if checkpoint_path is not None:
        checkpoint.restore(checkpoint_path)

    # summaries
    summary_writer = tf.contrib.summary.create_file_writer(
        args.job_dir, max_queue=100, flush_millis=5 * 60 * 1000)
    summary_writer.set_as_default()

    # rollouts
    rollout = Rollout(env, max_episode_steps=env.spec.max_episode_steps)
    random_strategy = RandomStrategy(action_space=env.action_space)

    # data
    with tf.device('cpu:0'):
        # train data
        states, actions, rewards, next_states, weights = rollout(
            random_strategy, episodes=params.episodes)

        dataset_train = tf.data.Dataset.from_tensor_slices(
            (states, actions, next_states, weights))
        dataset_train = dataset_train.batch(params.batch_size)
        dataset_train = dataset_train.prefetch(params.episodes)

        # eval data
        states, actions, rewards, next_states, weights = rollout(
            random_strategy, episodes=params.eval_episodes)

        dataset_eval = tf.data.Dataset.from_tensor_slices(
            (states, actions, next_states, weights))
        dataset_eval = dataset_eval.batch(params.batch_size)
        dataset_eval = dataset_eval.prefetch(params.episodes)

    for epoch in range(params.epochs):
        print('epoch', epoch)

        for states, actions, next_states, weights in dataset_train:
            print('global_step', global_step.numpy())
            with tf.GradientTape() as tape:
                # forward passes
                states_embed = state_embedding(states)
                next_states_embed = state_embedding(next_states)

                next_states_embed_pred = forward_model(
                    states_embed,
                    action_embedding(actions),
                    training=True,
                    reset_state=True)
                actions_pred = inverse_model(
                    states_embed,
                    next_states_embed_pred,
                    training=True,
                    reset_state=True)

                forward_loss = tf.losses.mean_squared_error(
                    predictions=next_states_embed_pred,
                    labels=next_states_embed,
                    weights=weights[..., None])
                inverse_loss = tf.losses.mean_squared_error(
                    predictions=actions_pred,
                    labels=actions,
                    weights=weights[..., None])
                loss = forward_loss + inverse_loss

            # optimization
            trainable_variables = (forward_model.trainable_variables +
                                   inverse_model.trainable_variables +
                                   state_embedding.trainable_variables +
                                   action_embedding.trainable_variables)
            grads = tape.gradient(loss, trainable_variables)
            if params.grad_clipping is not None:
                grads_clipped, _ = tf.clip_by_global_norm(
                    grads, params.grad_clipping)
            grads_and_vars = zip(grads_clipped, trainable_variables)
            optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            with tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar('loss/train', loss)
                tf.contrib.summary.scalar('forward_loss/train', forward_loss)
                tf.contrib.summary.scalar('inverse_loss/train', inverse_loss)
                tf.contrib.summary.scalar('gradient_norm',
                                          tf.global_norm(grads))
                tf.contrib.summary.scalar('gradient_norm/clipped',
                                          tf.global_norm(grads_clipped))

        for states, actions, next_states, weights in dataset_eval:
            states_embed = state_embedding(states)
            next_states_embed = state_embedding(next_states)

            next_states_embed_pred = forward_model(
                states_embed,
                action_embedding(actions),
                training=False,
                reset_state=True)
            actions_pred = inverse_model(
                states_embed,
                next_states_embed_pred,
                training=False,
                reset_state=True)

            forward_loss = tf.losses.mean_squared_error(
                predictions=next_states_embed_pred,
                labels=next_states_embed,
                weights=weights[..., None])
            inverse_loss = tf.losses.mean_squared_error(
                predictions=actions_pred,
                labels=actions,
                weights=weights[..., None])
            loss = forward_loss + inverse_loss

            with tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar('loss/eval', loss)
                tf.contrib.summary.scalar('forward_loss/eval', forward_loss)
                tf.contrib.summary.scalar('inverse_loss/eval', inverse_loss)

    # save checkpoint
    checkpoint_prefix = os.path.join(args.job_dir, 'ckpt')
    checkpoint.save(file_prefix=checkpoint_prefix)


if __name__ == '__main__':
    main()
