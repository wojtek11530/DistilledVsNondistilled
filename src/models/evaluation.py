import logging
import os
import sys
import time
from datetime import timedelta
from typing import Tuple, Dict, Any

import numpy as np
import torch
from sklearn.metrics import classification_report, accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel

from src.data.data_processing import get_num_labels, get_task_dataset
from src.models.utils import result_to_textfile, dictionary_to_json
from src.settings import MODELS_FOLDER

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)


def test_model(model_name: str, task_name: str, data_dir: str, batch_size: int = 32, max_seq_length: int = 512):
    output_dir = os.path.join(MODELS_FOLDER, model_name, task_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_labels = get_num_labels(task_name)

    # LOADING THE BEST MODEL
    model = AutoModelForSequenceClassification.from_pretrained(
        output_dir,
        num_labels=num_labels
    )
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    logger.info(f"Best model from {output_dir} loaded.")

    test_dataset = get_task_dataset(task_name, set_name='test', tokenizer=tokenizer,
                                    raw_data_dir=data_dir, max_seq_length=max_seq_length)
    logger.info("Test dataset loaded.")
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logger.info("\n***** Running evaluation on test dataset *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", batch_size)

    eval_start_time = time.monotonic()
    result, y_logits, y_true = evaluate(model, test_dataloader)
    eval_end_time = time.monotonic()

    diff = timedelta(seconds=eval_start_time - eval_end_time)
    diff_seconds = int(diff.total_seconds())
    result['eval_time'] = diff_seconds
    result_to_textfile(result, os.path.join(output_dir, "test_results.txt"))

    y_pred = np.argmax(y_logits, axis=1)
    print('\n\t**** Classification report ****\n')
    print(classification_report(y_true, y_pred))

    report = classification_report(y_true, y_pred, output_dict=True)
    report['eval_time'] = diff_seconds
    dictionary_to_json(report, os.path.join(output_dir, "test_results.json"))


def evaluate(model: PreTrainedModel, eval_dataloader: DataLoader) \
        -> Tuple[Dict[Any, Any], np.ndarray, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_loss = 0.0
    nb_eval_steps = 0
    all_logits = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)

        tmp_eval_loss = outputs.loss
        logits = outputs.logits
        eval_loss += tmp_eval_loss.mean().item()

        nb_eval_steps += 1
        if all_logits is None:
            all_logits = logits.detach().cpu().numpy()
            out_label_ids = batch['labels'].detach().cpu().numpy()
        else:
            all_logits = np.append(all_logits, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, batch['labels'].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps

    results = compute_metrics((all_logits, out_label_ids))
    results['eval_loss'] = eval_loss
    return results, all_logits, out_label_ids


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=preds)
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    return {"accuracy": accuracy, "f1": f1}
