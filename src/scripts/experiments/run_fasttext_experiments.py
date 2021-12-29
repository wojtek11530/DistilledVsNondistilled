import logging
import os
import sys

from src.settings import DATA_FOLDER, PROJECT_FOLDER, MODELS_FOLDER

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)

data_dir = os.path.join('data', 'multiemo2')

dim = 300

tasks_to_model = {
    'multiemo_en_all_sentence': 'cc.en.300',
    'multiemo_pl_all_sentence': 'kgr10.plain.skipgram.dim300.neg10'
}


def main():
    cmd = 'cd ' + PROJECT_FOLDER
    run_process(cmd)

    if not os.path.exists(os.path.join(DATA_FOLDER, 'multiemo2')):
        logger.info("Downloading Multiemo data")
        cmd = 'python3 -m src.scripts.download_dataset'
        run_process(cmd)
        logger.info("Downloading finished")

    if not os.path.exists(os.path.join(MODELS_FOLDER, 'fasttext')):
        logger.info("Downloading Fasttext models")
        cmd = 'python3 -m src.fasttext_models.download_models'
        run_process(cmd)
        logger.info("Downloading finished")

    for task, model in tasks_to_model.items():
        cmd = 'python3 -m src.scripts.run_fasttext_training '
        options = [
            '--model_name', model,
            '--data_dir', data_dir,
            '--task_name', task,
            '--dim', str(dim)
        ]
        cmd += ' '.join(options)

        logger.info(f"Training {model} for {task}")
        run_process(cmd)

        cmd = f'python3 -m src.scripts.gather_fasttext_results --task_name {task}'
        logger.info(f"Gathering fasttext results to csv for {task}")
        run_process(cmd)


def run_process(proc):
    os.system(proc)


if __name__ == '__main__':
    main()
