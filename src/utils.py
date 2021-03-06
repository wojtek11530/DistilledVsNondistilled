import json
import logging
import os
import sys

from src.settings import MODELS_FOLDER_2

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)


def result_to_text_file(result: dict, file_name: str, verbose: bool = True) -> None:
    with open(file_name, "a") as writer:
        if verbose:
            logger.info("***** Eval results *****")

        for key in sorted(result.keys()):
            if verbose:
                logger.info(" %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

        writer.write("")


def dictionary_to_json(dictionary: dict, file_name: str):
    with open(file_name, "w") as f:
        json.dump(dictionary, f, indent=2)


def is_folder_empty(folder_name: str):
    if len([f for f in os.listdir(folder_name) if not f.startswith('.')]) == 0:
        return True
    else:
        return False


def get_immediate_subdirectories(directory: str):
    return [os.path.join(directory, name) for name in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, name))]


def manage_output_dir(model_name: str, task_name: str) -> str:
    output_dir = os.path.join(MODELS_FOLDER_2, model_name, task_name)
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