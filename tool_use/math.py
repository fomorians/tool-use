import numpy as np
import tensorflow as tf


def normalize(x, low, high):
    """
    Normalize to standard distribution.
    """
    mean = (high + low) / 2
    stddev = (high - low) / 2
    stddev = np.where(np.isclose(stddev, 0.0), 1.0, stddev)
    x = (x - mean) / stddev
    x = tf.check_numerics(x, 'normalize')
    return x


def denormalize(x, low, high):
    """
    Denormalize to original distribution.
    """
    mean = (high + low) / 2
    stddev = (high - low) / 2
    stddev = np.where(np.isclose(stddev, 0.0), 1.0, stddev)
    x = (x * stddev) + mean
    x = tf.check_numerics(x, 'denormalize')
    return x
