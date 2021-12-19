import logging
import os
import sys

from src.settings import DATA_FOLDER, PROJECT_FOLDER

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)

data_dir = os.path.join('data', 'multiemo2')

REP_NUM = 4

max_seq_length = 512
batch_size = 8
num_train_epochs = 4
learning_rate = 5e-5
weight_decay = 0.01
warmup_steps = 0

tasks_to_models = {
    'multiemo_en_all_text':
        ['bert-base-uncased',
         'distilbert-base-uncased',
         'huawei-noah/TinyBERT_General_4L_312D',
         'huawei-noah/TinyBERT_General_6L_768D',
         'microsoft/xtremedistil-l6-h256-uncased'
         ]
}


def main():
    print(PROJECT_FOLDER)
    os.chdir(PROJECT_FOLDER)

    if not os.path.exists(os.path.join(DATA_FOLDER, 'multiemo2')):
        logger.info("Downloading Multiemo data")
        cmd = 'python3 -m src.scripts.download_dataset'
        run_process(cmd)
        logger.info("Downloading finished")

    for task, models in tasks_to_models.items():
        for model in models:
            for i in range(REP_NUM):
                cmd = 'python3 -m src.scripts.run_training '
                options = [
                    '--model_name', model,
                    '--data_dir', data_dir,
                    '--task_name', task,
                    '--batch_size', str(batch_size),
                    '--num_train_epochs', str(num_train_epochs),
                    '--learning_rate', str(learning_rate),
                    '--weight_decay', str(weight_decay),
                    '--warmup_steps', str(warmup_steps),
                    '--max_seq_length', str(max_seq_length),
                    '--do_lower_case'
                ]
                cmd += ' '.join(options)

                logger.info(f"Training {model} for {task}")
                run_process(cmd)

        cmd = f'python3 -m src.scripts.gather_results --task_name {task}'
        logger.info(f"Gathering results to csv for {task}")
        run_process(cmd)


def run_process(proc):
    os.system(proc)


if __name__ == '__main__':
    main()
