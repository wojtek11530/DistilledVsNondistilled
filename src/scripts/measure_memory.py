import argparse
import json
import os
from typing import Any, Dict

import GPUtil
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification

from src.data.data_processing import get_num_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",
                        default=None,
                        type=str,
                        help="The direction of the fine-tuned model.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")

    args = parser.parse_args()

    model_dir = args.model_dir
    task_name = args.task_name

    num_labels = get_num_labels(task_name)

    # LOADING THE BEST MODEL
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        num_labels=num_labels
    )
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(device))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(device) / 1024 ** 2, 1), 'MB')
        print('Cached:   ', round(torch.cuda.memory_reserved(device) / 1024 ** 2, 1), 'MB')

        GPUs = GPUtil.getGPUs()
        used_gpu = GPUs[torch.cuda.current_device()]
        print('Used:', used_gpu.memoryUsed)

    input("Press Enter to continue...")


if __name__ == '__main__':
    main()
