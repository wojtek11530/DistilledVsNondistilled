import logging
import os
import sys

from src.settings import PROJECT_FOLDER, DATA_FOLDER

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)

data_dir = os.path.join('data', 'multiemo2')

REP_NUM = 1

batch_size = 16
num_train_epochs = 50
learning_rate = 5e-3
weight_decay = 0.01
dropout = 0.1

tasks = ['multiemo_en_all_sentence', 'multiemo_en_all_text']


def main():
    os.chdir(PROJECT_FOLDER)

    if not os.path.exists(os.path.join(DATA_FOLDER, 'multiemo2')):
        logger.info("Downloading Multiemo data")
        cmd = 'python3 -m src.scripts.download_dataset'
        run_process(cmd)
        logger.info("Downloading finished")

    for task_name in tasks:
        for i in range(REP_NUM):
            cmd = 'python3 -m src.scripts.run_labse_mlp_training '
            options = [
                '--data_dir', data_dir,
                '--task_name', task_name,
                '--batch_size', str(batch_size),
                '--epochs', str(num_train_epochs),
                '--learning_rate', str(learning_rate),
                '--dropout', str(dropout),
                '--weight_decay', str(weight_decay),
                '--do_test'
            ]
            cmd += ' '.join(options)

            logger.info(f"Training LaBSE for {task_name}")
            run_process(cmd)

        cmd = f'python3 -m src.scripts.gather_labse_results --task_name {task_name}'
        logger.info(f"Gathering results to csv for {task_name}")
        run_process(cmd)


def run_process(proc):
    os.system(proc)


if __name__ == '__main__':
    main()
