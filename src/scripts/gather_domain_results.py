import argparse
import glob
import json
import os
from typing import Any, Dict, List

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification

from src.data.data_processing import get_num_labels
from src.settings import DATA_FOLDER, MODELS_FOLDER_2

models = [
    'bert-base-uncased',
    'distilbert-base-uncased',
    'TinyBERT_General_4L_312D',
    'TinyBERT_General_6L_768D',
    'xtremedistil-l6-h256-uncased'
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_level",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task level: either 'text' or 'sentence'.")

    args = parser.parse_args()

    task_level = args.task_level

    if task_level not in ['sentence', 'text']:
        raise ValueError('task_level must be text or sentence')

    models_subdirectories = [x[0] for x in os.walk(MODELS_FOLDER_2)]
    models_subdirectories = filter(lambda subdir: task_level in subdir and
                                                  any([m in subdir for m in models]),
                                   models_subdirectories)
    models_subdirectories = sorted(models_subdirectories)

    data = list()
    for subdirectory in tqdm(models_subdirectories):
        if task_level in subdirectory and any([m in subdirectory for m in models]):
            data_dict_list = gather_results(subdirectory)
            for data_dict in data_dict_list:
                data.append(data_dict)

    df = pd.DataFrame(data)
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    df.to_csv(os.path.join(DATA_FOLDER, 'domain-results-' + task_level + '.csv'), index=False)


def gather_results(ft_model_dir: str) -> List[Dict[str, Any]]:
    data_from_dir = list()
    with open(os.path.join(ft_model_dir, 'training_params.json')) as json_file:
        training_data_dict = json.load(json_file)

    model_size = os.path.getsize(os.path.join(ft_model_dir, 'pytorch_model.bin'))
    # LOADING THE BEST MODEL
    num_labels = get_num_labels(training_data_dict['task_name'])
    model = AutoModelForSequenceClassification.from_pretrained(
        ft_model_dir,
        num_labels=num_labels
    )
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    memory_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    memory_buffers = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    memory_used = memory_params + memory_buffers  # in bytes

    parameters_num = 0
    for n, p in model.named_parameters():
        parameters_num += p.nelement()

    for json_file_path in glob.glob(f"{ft_model_dir}/test_results*.json"):
        with open(json_file_path) as json_file:
            test_data = json.load(json_file)
            [test_data_dict] = pd.json_normalize(test_data, sep='_').to_dict(orient='records')

        data = training_data_dict.copy()  # start with keys and values of x
        data.update(test_data_dict)

        data['model_size'] = model_size
        data['memory'] = memory_used
        data['parameters'] = parameters_num
        data['name'] = os.path.basename(ft_model_dir)

        result_filename = os.path.basename(json_file_path)
        if result_filename == 'test_results.json':
            data['eval_task_name'] = training_data_dict['task_name']
        else:
            eval_task_name = result_filename.split('test_results_')[-1].split('.')[0]
            data['eval_task_name'] = eval_task_name

        data_from_dir.append(data)

    return data_from_dir


if __name__ == '__main__':
    main()
