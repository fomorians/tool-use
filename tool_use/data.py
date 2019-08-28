import tensorflow as tf


def create_dataset(data, batch_size=None, epochs=None):
    """
    Create a dataset with the given data, batch size and number of epochs.
    """
    with tf.device("/cpu:0"):
        dataset = tf.data.Dataset.from_tensor_slices(data)
        if batch_size is not None:
            dataset = dataset.batch(batch_size, drop_remainder=True)
        if epochs is not None:
            dataset = dataset.repeat(epochs)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
