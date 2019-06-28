import os

import numpy as np
from tqdm import tqdm
import tensorflow as tf


metrics = [
    'episodic_rewards/eval/TrapTube-v0',
    'episodic_rewards/eval/PerceptualTrapTube-v0',
    'episodic_rewards/eval/StructuralTrapTube-v0',
    'episodic_rewards/eval/SymbolicTrapTube-v0',
    'episodic_rewards/eval/PerceptualStructuralTrapTube-v0',
    'episodic_rewards/eval/PerceptualSymbolicTrapTube-v0',
    'episodic_rewards/eval/StructuralSymbolicTrapTube-v0',
    'episodic_rewards/eval/PerceptualStructuralSymbolicTrapTube-v0',
]

keys = [
    'all',
    'base',
    'P',
    'St',
    'Sy',
    'PSt',
    'PSy',
    'StSy',
    'PStSy',
]

results = {
    'PPO': {}, 'PPO + ICM': {}
}
for i in tqdm(range(5)):
    job_root_dir = f'gs://tool-use-jobs/PerceptualStructuralSymbolicTrapTube-v0/{i}/'
    query_tag = 'episodic_rewards/train'
    job_dirs = tf.io.gfile.listdir(job_root_dir)
    for job_dir in job_dirs:
        event_files = tf.io.gfile.glob(os.path.join(job_root_dir, job_dir, 'events.out.tfevents.*'))
        for event_file in event_files:

            if 'experiment-4-ppo-500-iters/' in event_file:
                d = {m: [] for m in metrics}
                for e in tf.compat.v1.train.summary_iterator(event_file):
                    for v in e.summary.value:
                        if v.tag in metrics:
                            d[v.tag].append(
                                tf.io.decode_raw(
                                    input_bytes=v.tensor.tensor_content,
                                    out_type=tf.float32,
                                ).numpy()[0],
                            )

                x = np.array([d[key] for key in d.keys()])

                if i == 0:
                    for key in keys:
                        results['PPO'][key] = []

                results['PPO']['all'].append(x.mean(0).max())
                argmax = np.argmax(x.mean(0))

                for j in range(len(metrics)):
                    results['PPO'][keys[j + 1]].append(d[metrics[j]][argmax])

            elif 'experiment-4-ppo-500-iters-no-l2rl/' in event_file:
                d = {m: [] for m in metrics}
                for e in tf.compat.v1.train.summary_iterator(event_file):
                    for v in e.summary.value:
                        if v.tag in metrics:
                            d[v.tag].append(
                                tf.io.decode_raw(
                                    input_bytes=v.tensor.tensor_content,
                                    out_type=tf.float32,
                                ).numpy()[0],
                            )

                x = np.array([d[key] for key in d.keys()])

                if i == 0:
                    for key in keys:
                        results['PPO + ICM'][key] = []

                results['PPO + ICM']['all'].append(x.mean(0).max())
                argmax = np.argmax(x.mean(0))

                for j in range(len(metrics)):
                    results['PPO + ICM'][keys[j + 1]].append(d[metrics[j]][argmax])

for key in results:
    print(key)
    print()
    for key2 in results[key]:
        x = results[key][key2]
        print(f'{key2}, {x}, {np.mean(x):.3f}, {np.std(x):.3f}')

    print()
