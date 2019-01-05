import pyoneer as pynr
import tensorflow as tf


def total_reward(rewards, weights=1.0):
    total_reward = tf.tile(
        tf.reduce_sum(rewards, axis=-1, keep_dims=True), [1, rewards.shape[1]])
    total_reward = total_reward * weights
    total_reward = tf.check_numerics(total_reward, 'total_reward')
    total_reward = tf.stop_gradient(total_reward)
    return total_reward


def post_action_reward(rewards, weights=1.0):
    post_action_reward = tf.reverse(
        tf.transpose(
            tf.scan(lambda agg, cur: cur + agg,
                    tf.transpose(tf.reverse(rewards * weights, [1]), [1, 0]),
                    tf.zeros_like(rewards[:, -1]), 1, False), [1, 0]), [1])

    post_action_reward = post_action_reward * weights
    post_action_reward = tf.check_numerics(post_action_reward,
                                           'post_action_reward')
    post_action_reward = tf.stop_gradient(post_action_reward)
    return post_action_reward


def baseline_reward(rewards, values, weights=1.0):
    baseline_reward = tf.reverse(
        tf.transpose(
            tf.scan(lambda agg, cur: cur + agg,
                    tf.transpose(tf.reverse(rewards * weights, [1]), [1, 0]),
                    tf.zeros_like(rewards[:, -1]), 1, False), [1, 0]), [1])

    baseline_reward = baseline_reward - values
    baseline_reward = baseline_reward * weights
    baseline_reward = tf.check_numerics(baseline_reward, 'baseline_reward')
    baseline_reward = tf.stop_gradient(baseline_reward)
    return baseline_reward


def discounted_rewards(rewards, discount_factor=0.99, weights=1.0):
    """
    Compute discounted rewards.
    """
    returns = tf.reverse(
        tf.transpose(
            tf.scan(lambda agg, cur: cur + discount_factor * agg,
                    tf.transpose(tf.reverse(rewards * weights, [1]), [1, 0]),
                    tf.zeros_like(rewards[:, -1]), 1, False), [1, 0]), [1])

    returns = returns * weights
    returns = tf.check_numerics(returns, 'returns')
    returns = tf.stop_gradient(returns)
    return returns


def batched_index(values, indices):
    mask = tf.one_hot(indices, values.shape[-1], dtype=values.dtype)
    return tf.reduce_sum(values * mask, axis=-1)


def generalized_advantages(rewards,
                           values,
                           discount_factor=0.99,
                           lambda_factor=0.95,
                           weights=1.0,
                           normalize=True):
    """
    Compute generalized advantage for policy optimization. Equation 11 and 12.
    """
    sequence_lengths = tf.reduce_sum(weights, axis=1)
    last_steps = tf.cast(sequence_lengths - 1, tf.int32)
    bootstrap_values = batched_index(values, last_steps)

    values_next = tf.concat([values[:, 1:], bootstrap_values[:, None]], axis=1)

    deltas = rewards + discount_factor * values_next - values

    advantages = tf.reverse(
        tf.transpose(
            tf.scan(
                lambda agg, cur: cur + discount_factor * lambda_factor * agg,
                tf.transpose(tf.reverse(deltas * weights, [1]), [1, 0]),
                tf.zeros_like(deltas[:, -1]), 1, False), [1, 0]), [1])

    if normalize:
        advantages = pynr.math.weighted_moments_normalize(
            advantages, weights=weights)

    advantages = advantages * weights
    advantages = tf.check_numerics(advantages, 'advantages')
    advantages = tf.stop_gradient(advantages)
    return advantages
