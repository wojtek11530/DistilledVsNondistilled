import logging
import os
import sys
import time
from datetime import timedelta
from typing import Tuple

import fasttext

from src.data.data_processing import get_task_dataset_dir
from src.fasttext_models.evaluation import test_model
from src.settings import MODELS_FOLDER
from src.utils import dictionary_to_json, is_folder_empty

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)

PYTORCH_LOOP_TRAINING = True


def train_model(model_name: str, task_name: str, data_dir: str, dim: int = 300, do_test: bool = True):
    output_dir = manage_output_dir(model_name, task_name)
    output_dir_quantized = manage_output_dir(model_name, task_name, quantized=True)

    train_set_dir = get_task_dataset_dir(task_name, data_dir, 'train')
    dev_set_dir = get_task_dataset_dir(task_name, data_dir, 'dev')

    # Training FT model
    model, training_duration = train_fastext_model(model_name, task_name, output_dir, dim, train_set_dir, dev_set_dir)

    # Quantization FT model
    quantize_fasttext_model(model, model_name, task_name, output_dir_quantized, train_set_dir, training_duration)

    if do_test:
        test_model(output_dir, task_name, data_dir)
        test_model(output_dir_quantized, task_name, data_dir)


def train_fastext_model(model_name: str, task_name: str, output_dir: str,
                        dim: int, train_set_dir: str, dev_set_dir: str) -> Tuple[fasttext.FastText, float]:
    model_vec_file = os.path.join(MODELS_FOLDER, 'fasttext', model_name + '.vec')
    # Training
    training_start_time = time.monotonic()
    model = fasttext.train_supervised(input=train_set_dir, dim=dim, pretrainedVectors=model_vec_file,
                                      autotuneValidationFile=dev_set_dir)
    training_end_time = time.monotonic()
    # Saving model
    model.save_model(os.path.join(output_dir, 'model.bin'))

    diff = timedelta(seconds=training_end_time - training_start_time)
    training_duration = diff.total_seconds()

    training_parameters = {'model_name': model_name, 'task_name': task_name, 'training_time': training_duration}
    output_training_params_file = os.path.join(output_dir, "training_params.json")
    dictionary_to_json(training_parameters, output_training_params_file)
    return model, training_duration


def quantize_fasttext_model(model: fasttext.FastText, model_name: str, task_name: str,
                            output_dir: str, train_set_dir: str, training_duration: float):
    # Quantization process
    quantization_start_time = time.monotonic()
    model = model.quantize(input=train_set_dir, retrain=True)
    quantization_end_time = time.monotonic()
    # Saving model
    model.save_model(os.path.join(output_dir, 'model.bin'))

    diff = timedelta(seconds=quantization_end_time - quantization_start_time)
    quantization_duration = diff.total_seconds()

    quantization_parameters = {'model_name': model_name, 'task_name': task_name,
                               'training_time': training_duration + quantization_duration}
    output_quantization_params_file = os.path.join(output_dir, "training_params.json")
    dictionary_to_json(quantization_parameters, output_quantization_params_file)


def manage_output_dir(model_name: str, task_name: str, quantized: bool = False) -> str:
    if quantized:
        model_name += '.quant'

    output_dir = os.path.join(MODELS_FOLDER, 'fasttext', model_name, task_name)
    run = 1
    while os.path.exists(output_dir + '-run-' + str(run)):
        if is_folder_empty(output_dir + '-run-' + str(run)):
            logger.info('folder exist but empty, use it as output')
            break
        logger.info(output_dir + '-run-' + str(run) + ' exist, trying next')
        run += 1
    output_dir += '-run-' + str(run)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir
