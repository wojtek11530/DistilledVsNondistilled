import logging
import os
import sys

from src.settings import DATA_FOLDER, PROJECT_FOLDER, MODELS_FOLDER_2

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)

data_dir = os.path.join('data', 'multiemo2')

REP_NUM = 5

max_seq_length = 256
batch_size = 8
num_train_epochs = 4
learning_rate = 5e-5
weight_decay = 0.01
warmup_steps = 0

mode_level = 'text'

models = [
    'bert-base-uncased',
    'distilbert-base-uncased',
    'huawei-noah/TinyBERT_General_4L_312D',
    'huawei-noah/TinyBERT_General_6L_768D',
    'microsoft/xtremedistil-l6-h256-uncased'
]

domains = ['reviews']

def main():
    print(PROJECT_FOLDER)
    os.chdir(PROJECT_FOLDER)

    for model in models:
        model_basename = manage_model_name(model)
        for domain in domains:
            task_name = f'multiemo_en_{domain}_{mode_level}'

            models_subdirectories = sorted([x[0] for x in os.walk(MODELS_FOLDER_2)])
            for subdirectory in models_subdirectories:
                if task_name in subdirectory and model_basename in subdirectory:
                    cmd = 'python3 -m src.scripts.run_evaluation '
                    options = [
                        '--model_dir', subdirectory,
                        '--data_dir', data_dir,
                        '--task_name', task_name,
                        '--batch_size', str(batch_size),
                        '--max_seq_length', str(max_seq_length),
                        '--do_lower_case'
                    ]
                    cmd += ' '.join(options)
                    logger.info(f"Evaluation model from {subdirectory} for {task_name}")
                    run_process(cmd)


def manage_model_name(name: str) -> str:
    spl = name.split('/')
    if len(spl) == 2:
        return spl[1]
    else:
        return name


def run_process(proc):
    os.system(proc)


if __name__ == '__main__':
    main()
