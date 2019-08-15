import os
import argparse

import numpy as np
from tqdm import tqdm
from cv2 import resize
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from tool_use.params import HyperParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--env-name")
    args = parser.parse_args()
    print(args)

    # params
    params_path = os.path.join(args.job_dir, "params.json")
    params = HyperParams.load(params_path)
    save_dir = args.job_dir.replace("gs://tool-use-jobs", "jobs")
    print(params)

    for env_name in params.eval_env_names:
        if args.env_name is not None and env_name != args.env_name:
            continue

        thumbnail = np.zeros([80 * 32, 80 * 32, 3])

        hidden_states = []
        actions = []
        timesteps = []
        path = os.path.join(save_dir, "transition_data", env_name)
        for i in tqdm(range(params.episodes_eval)):
            transition = np.load(os.path.join(path, f"{i}.npz"))
            for j in range(transition["hidden_states"].shape[0]):
                hidden_states.append(transition["hidden_states"][j])
                action = transition["actions"][j]
                action = action[0] * 4 + action[1]
                actions.append(action)
                timesteps.append(j)                

                image = transition["images"][j] / 255
                image = resize(image, dsize=(32, 32))
                thumbnail[
                    ((i * 50 + j) % 80) * 32 : ((i * 50 + j) % 80) * 32 + 32,
                    ((i * 50 + j) // 80) * 32 : ((i * 50 + j) // 80) * 32 + 32,
                ] = image
            
        hidden_states = np.array(hidden_states)

        # Hidden state vectors.
        path = os.path.join(
            save_dir, "hidden_states_tsv", env_name, "hidden_states.tsv"
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savetxt(path, hidden_states, delimiter="\t")

        # Thumbnail images of environment observations.
        path = os.path.join(save_dir, "hidden_states_tsv", env_name, "thumbnail.jpg")
        plt.imsave(path, thumbnail)

        # Metadata (color by).
        path = os.path.join(save_dir, "hidden_states_tsv", env_name, "metadata.tsv")
        with open(path, "w") as f:
            f.write("Index\tAction\tTimestep\n")
            for index, (action, timestep) in enumerate(zip(actions, timesteps)):
                f.write("%d\t%d\t%d\n" % (index, action, timestep))

        # Tensorboard projector configuration file.
        path = os.path.join(
            save_dir, "hidden_states_tsv", env_name, "projector_config.pbtxt"
        )
        with open(path, mode="w") as f:
            f.write(
                "embeddings {\n"
                "  tensor_path: 'hidden_states.tsv'\n"
                "  sprite {\n"
                "    image_path: 'thumbnail.jpg'\n"
                "    single_image_dim: 32\n"
                "    single_image_dim: 32\n"
                "  }\n"
                "}"
            )


if __name__ == "__main__":
    main()
