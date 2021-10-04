import argparse
import json
import os
from typing import Any, Dict

import pandas as pd

from src.settings import DATA_FOLDER, MODELS_FOLDER


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")

    args = parser.parse_args()

    task_name = args.task_name

    models_folders = get_immediate_subdirectories(MODELS_FOLDER)
    data = []
    for model_folder in models_folders:
        task_models_directories = get_immediate_subdirectories(model_folder)
        for task_models_directory in task_models_directories:
            if task_name in task_models_directory:
                data_dict = gather_results(task_models_directory)
                data.append(data_dict)

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(DATA_FOLDER, 'results-' + task_name + '.csv'), index=False)


def get_immediate_subdirectories(directory: str):
    return [os.path.join(directory, name) for name in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, name))]


def gather_results(task_models_directory: str) -> Dict[str, Any]:
    with open(os.path.join(task_models_directory, 'training_params.json')) as json_file:
        training_data_dict = json.load(json_file)

    with open(os.path.join(task_models_directory, 'test_results.json')) as json_file:
        test_data = json.load(json_file)
        [test_data_dict] = pd.json_normalize(test_data, sep='_').to_dict(orient='records')

    data = training_data_dict.copy()  # start with keys and values of x
    data.update(test_data_dict)

    model_size = os.path.getsize(os.path.join(task_models_directory, 'pytorch_model.bin'))
    data['model_size'] = model_size
    data['name'] = os.path.basename(task_models_directory)
    return data


if __name__ == '__main__':
    main()
