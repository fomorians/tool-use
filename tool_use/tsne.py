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
        
        hidden_states = []
        path = os.path.join(save_dir, "transition_data", env_name)
        for i in tqdm(range(params.episodes_eval)):
            transition = np.load(os.path.join(path, f"{i}.npz"))
            for j in range(transition["hidden_states"].shape[0]):
                hidden_states.append(transition["hidden_states"][j])

        hidden_states = np.array(hidden_states)

        print("Fitting embedding...", end=" ")
        embedded = TSNE(n_components=2).fit_transform(hidden_states)
        x, y = embedded[:, 0], embedded[:, 1]
        print("Done.")

        fig, ax = plt.subplots(figsize=(9, 9))
        ax.set_title(f"TSNE for {env_name} task")
        ax.scatter(x, y, s=1)

        for idx, (x0, y0) in enumerate(tqdm(zip(x, y))):
            i = idx // 50
            j = idx % 50
            image = np.load(os.path.join(path, f"{i}.npz"))["images"][j]
            image = resize(image, dsize=(16, 16))
            ab = AnnotationBbox(OffsetImage(image), (x0, y0), frameon=False)
            ax.add_artist(ab)
        
        path = os.path.join(save_dir, "tsne", env_name, "tsne.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        plt.show()


if __name__ == "__main__":
    main()