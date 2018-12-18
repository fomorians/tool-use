import tensorflow as tf


def compute_returns(rewards, discount_factor=0.99, weights=1.0):
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


def compute_advantages(rewards,
                       values,
                       discount_factor=0.99,
                       lambda_factor=0.95,
                       weights=1.0,
                       normalize=True):
    """
    Compute generalized advantage for policy optimization. Equation 11 and 12.
    """
    values_next = tf.concat(
        [values[:, 1:], tf.zeros_like(values[:, -1:])], axis=1)

    delta = rewards + discount_factor * values_next - values

    advantages = tf.reverse(
        tf.transpose(
            tf.scan(
                lambda agg, cur: cur + discount_factor * lambda_factor * agg,
                tf.transpose(tf.reverse(delta * weights, [1]), [1, 0]),
                tf.zeros_like(delta[:, -1]), 1, False), [1, 0]), [1])

    if normalize:
        advantages_mean, advantages_variance = tf.nn.weighted_moments(
            advantages, axes=[0, 1], frequency_weights=weights, keep_dims=True)
        advantages_stddev = tf.sqrt(advantages_variance + 1e-6) + 1e-8
        advantages = (advantages - advantages_mean) / advantages_stddev

    advantages = advantages * weights
    advantages = tf.check_numerics(advantages, 'advantages')
    advantages = tf.stop_gradient(advantages)
    return advantages
