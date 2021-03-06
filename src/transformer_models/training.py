import logging
import os
import sys
import time
from datetime import timedelta
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AdamW, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase, Trainer,
    TrainingArguments, get_linear_schedule_with_warmup)

from src.data.data_processing import Dataset, SmartCollator, get_num_labels, get_task_dataset
from src.transformer_models.evaluation import compute_metrics, evaluate, test_model
from src.settings import MODELS_FOLDER_2
from src.utils import dictionary_to_json, result_to_text_file, manage_output_dir

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)

PYTORCH_LOOP_TRAINING = True


def train_model(model_name: str, task_name: str, data_dir: str, epochs: int, batch_size: int = 32,
                learning_rate: float = 5e-5, weight_decay: float = 0.01, warmup_steps: int = 0,
                max_seq_length: int = 512, do_lower_case: bool = True, do_test: bool = True):
    training_parameters = {'model_name': model_name, 'task_name': task_name, 'epochs': epochs, 'batch_size': batch_size,
                           'learning_rate': learning_rate, 'weight_decay': weight_decay, 'warmup_steps': warmup_steps,
                           'max_seq_length': max_seq_length, 'do_lower_case': do_lower_case}

    output_dir = manage_output_dir(model_name, task_name)
    output_training_params_file = os.path.join(output_dir, "training_params.json")

    num_labels = get_num_labels(task_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        cache_dir=os.path.join(MODELS_FOLDER_2, model_name)
    )
    logger.info(f"Model {model_name} loaded.")

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=os.path.join(MODELS_FOLDER_2, model_name),
                                              do_lower_case=do_lower_case)
    logger.info(f"Tokenizer {model_name} loaded.")

    logger.info(f"Loading datasets for task {task_name}")
    train_dataset = get_task_dataset(task_name, set_name='train', tokenizer=tokenizer,
                                     raw_data_dir=data_dir, max_seq_length=max_seq_length)
    logger.info("Train dataset loaded.")
    dev_dataset = get_task_dataset(task_name, set_name='dev', tokenizer=tokenizer,
                                   raw_data_dir=data_dir, max_seq_length=max_seq_length)
    logger.info("Dev dataset loaded.")

    training_start_time = time.monotonic()
    if PYTORCH_LOOP_TRAINING:
        train_with_pytorch_loop(model, tokenizer, train_dataset, dev_dataset,
                                output_dir, epochs, batch_size, learning_rate,
                                warmup_steps, weight_decay)
    else:
        train_with_trainer(model, train_dataset, dev_dataset, output_dir,
                           epochs, batch_size, learning_rate, warmup_steps, weight_decay)

    training_end_time = time.monotonic()

    diff = timedelta(seconds=training_end_time - training_start_time)
    diff_seconds = diff.total_seconds()
    training_parameters['training_time'] = diff_seconds

    dictionary_to_json(training_parameters, output_training_params_file)

    if do_test:
        test_model(output_dir, task_name, data_dir, batch_size, max_seq_length, do_lower_case)


def train_with_pytorch_loop(
        model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase,
        train_dataset: Dataset, dev_dataset: Dataset,
        output_dir: str, epochs: int, batch_size: int, learning_rate: float,
        warmup_steps: int, weight_decay: float) \
        -> Tuple[int, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device: {}".format(device))
    model.to(device)

    collator = SmartCollator(tokenizer.pad_token_id)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collator.collate_batch,
                                  pin_memory=True, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=collator.collate_batch,
                                pin_memory=True, shuffle=False)

    num_training_steps = epochs * len(train_dataloader)
    optimizer = get_optimizer(model, learning_rate, weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=num_training_steps)

    logger.info("\n***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num epochs = %d", epochs)
    logger.info("  Batch size = %d", batch_size)
    logger.info("  Num steps = %d", num_training_steps)

    global_step = 0
    best_dev_acc = 0.0
    tr_loss = 0.0
    output_eval_file = os.path.join(output_dir, "eval_results.txt")
    model.zero_grad()

    for epoch in range(int(epochs)):
        model.train()
        for batch in tqdm(train_dataloader, f"Epoch {epoch + 1}: "):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            tr_loss += loss.item()

            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1

        avg_train_loss = tr_loss / len(train_dataloader)

        logger.info("\n***** Running evaluation *****")
        logger.info("  Epoch = {} iter {} step".format(epoch, global_step))
        logger.info("  Num examples = %d", len(dev_dataset))
        logger.info("  Batch size = %d", batch_size)

        result, _, _ = evaluate(model, dev_dataloader, device)
        result['global_step'] = global_step
        result['avg_train_loss'] = avg_train_loss

        result_to_text_file(result, output_eval_file)

        save_model = False
        if result['accuracy'] > best_dev_acc:
            best_dev_acc = result['accuracy']
            save_model = True

        if save_model:
            logger.info("\n***** Save model *****")
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)

        print()

    logger.info("Training finished.")

    if global_step > 0:
        return global_step, tr_loss / global_step
    else:
        return global_step, tr_loss


def get_optimizer(model: PreTrainedModel, learning_rate: float, weight_decay: float) -> AdamW:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    return optimizer


def train_with_trainer(model: PreTrainedModel, train_dataset: Dataset, dev_dataset: Dataset, output_dir: str,
                       epochs: int, batch_size: int, learning_rate: float, warmup_steps: int, weight_decay: float):
    # Define Trainer
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,  # number of warmup steps for learning rate scheduler
        weight_decay=weight_decay,  # strength of weight decay
        per_device_train_batch_size=batch_size,  # batch size per device during training
        per_device_eval_batch_size=batch_size,  # batch size for evaluation
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True
    )
    trainer = Trainer(
        model=model,  # the instantiated ???? Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=dev_dataset,  # evaluation dataset
        compute_metrics=compute_metrics
    )
    # Train pre-trained model
    logger.info("***** Running training *****")
    trainer.train()
    logger.info("Training finished.")
