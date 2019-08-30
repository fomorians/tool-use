import gym
import imageio
import tensorflow as tf
import pyoneer.rl as pyrl

from tool_use.batch_rollout import BatchRollout


def create_env(env_name):
    """
    Create an environment and apply observation and action processing.
    """

    def _create_env():
        env = gym.make(env_name)
        env = pyrl.wrappers.ObservationCoordinates(env)
        env = pyrl.wrappers.ObservationNormalization(env)
        env = pyrl.wrappers.MultiActionProbs(env)
        return env

    return _create_env


def collect_transitions(env_name, episodes, batch_size, policy, seed, render_mode=None):
    """
    Collect transitions from the environment.
    """
    with tf.device("/cpu:0"):
        env = pyrl.wrappers.Batch(create_env(env_name), batch_size=batch_size)
        env.seed(seed)
        rollout = BatchRollout(env)
        transitions = rollout(policy, episodes, render_mode=render_mode)
    return transitions


def flatten_transitions(transitions):
    # flatten episode and step dimensions of transitions
    transitions_flat = {}
    for key, val in transitions.items():
        transitions_flat[key] = val.reshape((-1,) + val.shape[2:])

    indices = transitions_flat["weights"] > 0.0

    # remove empty transitions
    for key, val in transitions_flat.items():
        transitions_flat[key] = val[indices]

    return transitions_flat


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


def save_images(job_dir, env_name, transitions):
    """
    Save images of rollouts.
    """
    timestamp = int(time.time())
    image_dir = os.path.join(job_dir, "results", env_name, str(timestamp))
    success_image_dir = os.path.join(image_dir, "success")
    failure_image_dir = os.path.join(image_dir, "failure")

    os.makedirs(success_image_dir, exist_ok=True)
    os.makedirs(failure_image_dir, exist_ok=True)

    for episode, episode_images in enumerate(transitions["images"]):
        rewards = np.sum(transitions["rewards"][episode], axis=-1)

        if rewards > 0:
            image_path = os.path.join(success_image_dir, "{}.gif".format(episode))
        else:
            image_path = os.path.join(failure_image_dir, "{}.gif".format(episode))

        episode_weights = transitions["weights"][episode]
        max_episode_steps = int(episode_weights.sum())
        imageio.mimwrite(
            image_path,
            episode_images[: max_episode_steps + 1],
            loop=1,
            fps=3,
            subrectangles=True,
        )
