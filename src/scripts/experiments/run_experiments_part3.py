import logging
import os
import sys

from src.settings import DATA_FOLDER, PROJECT_FOLDER

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)

data_dir = os.path.join('data', 'multiemo2')

max_seq_length = 512
batch_size = 16
num_train_epochs = 3
learning_rate = 5e-5
weight_decay = 0.01
warmup_steps = 0

tasks_to_models = {
    'multiemo_en_all_sentence':
        ['nreimers/MiniLM-L6-H384-uncased',
         'nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base',
         'nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large'
         ]
}


def main():
    cmd = 'cd ' + PROJECT_FOLDER
    run_process(cmd)

    if not os.path.exists(os.path.join(DATA_FOLDER, 'multiemo2')):
        logger.info("Downloading Multiemo data")
        cmd = 'python3 -m src.scripts.download_dataset'
        run_process(cmd)
        logger.info("Downloading finished")

    for task, models in tasks_to_models.items():
        for model in models:
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
                '--do_test'
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
