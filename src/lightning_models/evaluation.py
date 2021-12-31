import json
import os
import time
from datetime import timedelta
from typing import Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.data_processing import get_num_labels, get_labels
from src.data.labse_datamodule import LabseDataset
from src.lightning_models.mlp import MLPClassifier
from src.settings import MODELS_FOLDER_2
from src.utils import dictionary_to_json


def test_model(model_dir: str) -> None:
    with open(os.path.join(model_dir, 'training_params.json')) as json_file:
        hyperparams = json.load(json_file)

    task_name = hyperparams['task_name']

    num_labels = get_num_labels(task_name)
    labels_list = get_labels(task_name)

    model = MLPClassifier.load_from_checkpoint(
        checkpoint_path=os.path.join(model_dir, 'model.chkpt'),
        input_size=hyperparams['input_size'],
        hidden_size=hyperparams['hidden_size'],
        output_size=num_labels,
        learning_rate=hyperparams['learning_rate'],
        weight_decay=hyperparams['weight_decay'],
        dropout=hyperparams['dropout']
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    labse_model = SentenceTransformer("sentence-transformers/LaBSE",
                                      cache_folder=os.path.join(MODELS_FOLDER_2, 'LaBSE'))

    dataset = LabseDataset(
        task_name=task_name, raw_data_dir=hyperparams['data_dir'],
        set_name='test', embedder=labse_model
    )
    test_dataloader = DataLoader(dataset, batch_size=hyperparams['batch_size'], shuffle=False)

    eval_start_time = time.monotonic()
    y_logits, y_true = evaluate(model, test_dataloader, device)
    eval_end_time = time.monotonic()

    diff = timedelta(seconds=eval_end_time - eval_start_time)
    diff_seconds = diff.total_seconds()

    y_pred = np.argmax(y_logits, axis=1)
    print('\n\t**** Classification report ****\n')
    print(classification_report(y_true, y_pred, target_names=labels_list))

    report = classification_report(y_true, y_pred, target_names=labels_list, output_dict=True)
    report['eval_time'] = diff_seconds
    dictionary_to_json(report, os.path.join(model_dir, "test_results.json"))


def evaluate(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    model = model.eval()
    all_logits = None
    out_label_ids = None

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = (v.to(device) for v in batch)
            x, y_labels = batch
            logits = model(x)

            if all_logits is None:
                all_logits = logits.detach().cpu().numpy()
                out_label_ids = y_labels.detach().cpu().numpy()
            else:
                all_logits = np.append(all_logits, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, y_labels.detach().cpu().numpy(), axis=0)

    return all_logits, out_label_ids
