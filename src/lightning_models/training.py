import logging
import os
import sys
import time
from datetime import timedelta

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping

from src.data.data_processing import get_num_labels
from src.data.labse_datamodule import LabseDataModule
from src.lightning_models.evaluation import test_model
from src.lightning_models.mlp import MLPClassifier
from src.utils import dictionary_to_json, manage_output_dir

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)


def train_model(args):
    logger.info(f"Loading datasets for task {args.task_name}")
    datamodule = LabseDataModule(
        task_name=args.task_name,
        raw_data_dir=args.data_dir,
        batch_size=args.batch_size
    )
    logger.info("Datasets loaded.")

    num_labels = get_num_labels(args.task_name)

    evaluate = args.do_test
    training_parameters = vars(args)
    training_parameters.pop('do_test')

    output_dir = manage_output_dir('MLP-LaBSE', args.task_name)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=[EarlyStopping(monitor='val_loss', mode='min', patience=6, verbose=True)],
        gpus=1 if torch.cuda.is_available() else None
    )

    model = MLPClassifier(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        output_size=num_labels,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout
    )

    training_start_time = time.monotonic()
    trainer.fit(model, datamodule)
    training_end_time = time.monotonic()

    diff = timedelta(seconds=training_end_time - training_start_time)
    diff_seconds = diff.total_seconds()
    training_parameters['training_time'] = diff_seconds

    trainer.save_checkpoint(filepath=os.path.join(output_dir, 'model.chkpt'))

    output_training_params_file = os.path.join(output_dir, "training_params.json")
    dictionary_to_json(training_parameters, output_training_params_file)

    if evaluate:
        print('Run model evaluation')
        test_model(output_dir)
