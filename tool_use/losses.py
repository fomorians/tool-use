import tensorflow as tf


def policy_ratio_loss(log_probs,
                      log_probs_anchor,
                      advantages,
                      epsilon_clipping=0.2,
                      weights=1.0):
    log_probs_anchor = tf.stop_gradient(log_probs_anchor)
    advantages = tf.stop_gradient(advantages)

    ratio = tf.exp(log_probs - log_probs_anchor)
    surrogate1 = ratio * advantages
    surrogate2 = tf.clip_by_value(ratio, 1 - epsilon_clipping,
                                  1 + epsilon_clipping) * advantages
    losses = tf.minimum(surrogate1, surrogate2)

    loss = -tf.losses.compute_weighted_loss(losses=losses, weights=weights)
    loss = tf.check_numerics(loss, 'loss')
    return loss
