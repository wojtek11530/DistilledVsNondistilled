import argparse
import json
import os
from typing import Any, Dict

import GPUtil
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification

from src.data.data_processing import get_num_labels
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
        fine_tuned_model_directories = get_immediate_subdirectories(model_folder)
        for ft_model_dir in fine_tuned_model_directories:
            if task_name in ft_model_dir:
                data_dict = gather_results(ft_model_dir, task_name)
                data.append(data_dict)

    df = pd.DataFrame(data)
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    df.to_csv(os.path.join(DATA_FOLDER, 'results-' + task_name + '.csv'), index=False)


def get_immediate_subdirectories(directory: str):
    return [os.path.join(directory, name) for name in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, name))]


def gather_results(ft_model_dir: str, task_name: str) -> Dict[str, Any]:
    with open(os.path.join(ft_model_dir, 'training_params.json')) as json_file:
        training_data_dict = json.load(json_file)

    with open(os.path.join(ft_model_dir, 'test_results.json')) as json_file:
        test_data = json.load(json_file)
        [test_data_dict] = pd.json_normalize(test_data, sep='_').to_dict(orient='records')

    data = training_data_dict.copy()  # start with keys and values of x
    data.update(test_data_dict)

    model_size = os.path.getsize(os.path.join(ft_model_dir, 'pytorch_model.bin'))
    data['model_size'] = model_size

    num_labels = get_num_labels(task_name)

    # LOADING THE BEST MODEL
    model = AutoModelForSequenceClassification.from_pretrained(
        ft_model_dir,
        num_labels=num_labels
    )

    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if device.type == 'cuda':
        current_gpu = GPUtil.getGPUs()[torch.cuda.current_device()]
        data['gpu_memory_used'] = current_gpu.memoryUsed

    memory_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    memory_buffers = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    memory_used = memory_params + memory_buffers  # in bytes

    data['memory'] = memory_used

    parameters_num = 0
    for n, p in model.named_parameters():
        parameters_num += p.nelement()

    data['parameters'] = parameters_num
    data['name'] = os.path.basename(ft_model_dir)
    return data


if __name__ == '__main__':
    main()
