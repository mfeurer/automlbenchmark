import glob
import openml
import numpy as np
import os
import pandas as pd
import yaml
import tqdm

# Read in the available task
benchmarks = glob.glob(os.path.join('resources', 'benchmarks', '*yaml'))
tasks = []

for benchmark in benchmarks:
    with open(benchmark) as file:
        tasks.extend(yaml.load(file, Loader=yaml.FullLoader))

# Complement task with relevant information
for i, task in tqdm.tqdm(enumerate(tasks), total=len(tasks)):
    try:
        openml_task = openml.tasks.get_task(task['openml_task_id'])
        tasks[i]['dataset_id'] = openml_task.dataset_id
        data = openml.datasets.get_dataset(openml_task.dataset_id, download_data=False)
        tasks[i].update(data.qualities)
    except Exception as e:
        print(f"Failed on {task} due to {e}")

pd.DataFrame(tasks).to_csv('benchmark_info.csv')
