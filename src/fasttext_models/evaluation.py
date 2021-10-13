import logging
import os
import sys
import time
from datetime import timedelta
from typing import List, Tuple

import fasttext
from sklearn.metrics import classification_report

from src.data.data_processing import get_task_dataset_dir
from src.utils import dictionary_to_json

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)


def test_model(model_dir: str, task_name: str, data_dir: str):
    test_set_dir = get_task_dataset_dir(task_name, 'test', data_dir)
    texts, true_labels = extract_texts_and_labels(test_set_dir)

    # Loading model
    model_path = os.path.join(model_dir, 'model.bin')
    model = fasttext.load_model(model_path)
    logger.info(f"Loaded model from {model_path}")

    # Evaluation
    logger.info("\n***** Running evaluation on test dataset *****")
    eval_start_time = time.monotonic()
    predictions = model.predict(texts)
    eval_end_time = time.monotonic()

    preds = [label.split('__label__')[1] for sublist in predictions[0] for label in sublist]

    print('\n\t**** Classification report ****\n')
    print(classification_report(true_labels, preds))

    report = classification_report(true_labels, preds, output_dict=True)
    diff = timedelta(seconds=eval_end_time - eval_start_time)
    diff_seconds = diff.total_seconds()
    report['eval_time'] = diff_seconds
    dictionary_to_json(report, os.path.join(model_dir, "test_results.json"))


def extract_texts_and_labels(test_set_dir: str) -> Tuple[List[str], List[str]]:
    texts = []
    true_labels = []
    with open(test_set_dir, "r", encoding='UTF-8') as f:
        lines = f.read().splitlines()
        for (i, line) in enumerate(lines):
            split_line = line.split('__label__')
            text = split_line[0]
            label = split_line[1]
            texts.append(text)
            true_labels.append(label)
    return texts, true_labels
