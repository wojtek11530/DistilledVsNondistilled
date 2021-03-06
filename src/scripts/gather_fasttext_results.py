import argparse
import json
import os
from typing import Any, Dict

import pandas as pd

from src.settings import DATA_FOLDER, MODELS_FOLDER
from src.utils import get_immediate_subdirectories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")

    args = parser.parse_args()

    task_name = args.task_name

    fasttext_folder = os.path.join(MODELS_FOLDER, 'fasttext')
    models_folders = get_immediate_subdirectories(fasttext_folder)
    data = []
    for model_folder in models_folders:
        fine_tuned_model_directories = get_immediate_subdirectories(model_folder)
        for ft_model_dir in fine_tuned_model_directories:
            if task_name in ft_model_dir:
                data_dict = gather_results(ft_model_dir)
                data.append(data_dict)

    df = pd.DataFrame(data)
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    df.to_csv(os.path.join(DATA_FOLDER, 'fasttext_results-' + task_name + '.csv'), index=False)


def gather_results(ft_model_dir: str) -> Dict[str, Any]:
    with open(os.path.join(ft_model_dir, 'training_params.json')) as json_file:
        training_data_dict = json.load(json_file)

    with open(os.path.join(ft_model_dir, 'test_results.json')) as json_file:
        test_data = json.load(json_file)
        [test_data_dict] = pd.json_normalize(test_data, sep='_').to_dict(orient='records')

    data = training_data_dict.copy()  # start with keys and values of x
    data.update(test_data_dict)

    model_size = os.path.getsize(os.path.join(ft_model_dir, 'model.bin'))
    data['model_size'] = model_size
    data['name'] = os.path.basename(ft_model_dir)
    return data


if __name__ == '__main__':
    main()
