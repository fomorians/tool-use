import tensorflow as tf


def compute_improved_returns(advantages_norm, values, weights=1.0):
    """
    Compute improved returns.
    """
    return tf.stop_gradient((advantages_norm + values) * weights)


def compute_discounted_rewards(rewards, discount_factor=0.99, weights=1.0):
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


def compute_advantages(rewards,
                       values,
                       discount_factor=0.99,
                       lambda_factor=0.95,
                       weights=1.0):
    """
    Compute generalized advantage for policy optimization. Equation 11 and 12.
    """
    sequence_lengths = tf.reduce_sum(weights, axis=1)
    last_steps = tf.cast(sequence_lengths - 1, tf.int32)
    bootstrap_values = batched_index(values, last_steps)

    values_next = tf.concat([values[:, 1:], bootstrap_values[:, None]], axis=1)

    delta = rewards + discount_factor * values_next - values

    advantages = tf.reverse(
        tf.transpose(
            tf.scan(
                lambda agg, cur: cur + discount_factor * lambda_factor * agg,
                tf.transpose(tf.reverse(delta * weights, [1]), [1, 0]),
                tf.zeros_like(delta[:, -1]), 1, False), [1, 0]), [1])

    advantages = advantages * weights
    advantages = tf.check_numerics(advantages, 'advantages')
    advantages = tf.stop_gradient(advantages)
    return advantages
