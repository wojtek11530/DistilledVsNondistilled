import logging
import os
import sys

import numpy as np
import torch

from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader
from tqdm.auto import trange, tqdm
from transformers import AdamW, AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, \
    PreTrainedModel, PreTrainedTokenizerBase, Trainer, TrainingArguments

from src.data_processing import Dataset, get_num_labels, get_output_mode, get_task_dataset
from settings import MODELS_FOLDER

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
logger = logging.getLogger(__name__)


def train_model(model_name: str, task_name: str, data_dir: str, epochs: int, batch_size: int = 32,
                learning_rate: float = 5e-5, weight_decay: float = 0.01, warmup_steps: int = 0,
                max_seq_length: int = 512):
    output_dir = os.path.join(MODELS_FOLDER, model_name, task_name)

    num_labels = get_num_labels(task_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        cache_dir=os.path.join(MODELS_FOLDER, model_name)
    )
    logger.info(f"Model {model_name} loaded.")

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=os.path.join(MODELS_FOLDER, model_name))
    logger.info(f"Tokenizer {model_name} loaded.")

    logger.info(f"Loading datasets for task {task_name}")
    train_dataset = get_task_dataset(task_name, set_name='train', tokenizer=tokenizer,
                                     raw_data_dir=data_dir, max_seq_length=max_seq_length)
    logger.info("Train dataset loaded.")
    dev_dataset = get_task_dataset(task_name, set_name='dev', tokenizer=tokenizer,
                                   raw_data_dir=data_dir, max_seq_length=max_seq_length)
    logger.info("Dev dataset loaded.")

    return train_with_pytorch_loop(model, tokenizer, train_dataset, dev_dataset,
                                   output_dir, task_name, epochs, batch_size, learning_rate,
                                   warmup_steps, weight_decay)

    # train_with_trainer(batch_size, data_dir, dev_dataset, epochs, learning_rate, max_seq_length,
    # model, task_name, output_dir, tokenizer, train_dataset, warmup_steps, weight_decay)


def train_with_pytorch_loop(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase,
                            train_dataset: Dataset, dev_dataset: Dataset,
                            output_dir: str, task_name: str,
                            epochs: int, batch_size: int, learning_rate: float,
                            warmup_steps: int, weight_decay: float):
    output_mode = get_output_mode(task_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device: {}".format(device))
    model.to(device)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    num_training_steps = epochs * len(train_dataloader)

    optimizer = get_optimizer(model, learning_rate, weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=num_training_steps)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Batch size = %d", batch_size)
    logger.info("  Num steps = %d", num_training_steps)

    global_step = 0
    best_dev_acc = 0.0
    tr_loss = 0.0
    output_eval_file = os.path.join(output_dir, "eval_results.txt")
    model.zero_grad()

    training_iterator = trange(int(epochs), desc="Epoch")
    for epoch in training_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
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

        logger.info("***** Running evaluation *****")
        logger.info("  Epoch = {} iter {} step".format(epoch, global_step))
        logger.info("  Num examples = %d", len(dev_dataset))
        logger.info("  Batch size = %d", batch_size)

        result = evaluate(model, dev_dataloader, output_mode, device)
        result['global_step'] = global_step
        result['avg_train_loss'] = avg_train_loss

        result_to_file(result, output_eval_file)

        save_model = False
        if result['acc'] > best_dev_acc:
            best_dev_acc = result['acc']
            save_model = True

        if save_model:
            logger.info("***** Save model *****")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)

    logger.info("Training finished.")
    return global_step, tr_loss / global_step


def get_optimizer(model: PreTrainedModel, learning_rate: float, weight_decay: float):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    return optimizer


def evaluate(model: PreTrainedModel, eval_dataloader: DataLoader, output_mode: str, device: str):
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)

        tmp_eval_loss = outputs.loss
        logits = outputs.logits
        eval_loss += tmp_eval_loss.mean().item()

        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = batch['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, batch['labels'].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    if output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif output_mode == "regression":
        preds = np.squeeze(preds)

    results = compute_metrics((preds, out_label_ids))
    results['eval_loss'] = eval_loss
    return results


def result_to_file(result: dict, file_name: str):
    with open(file_name, "a") as writer:
        writer.write("")
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))


def train_with_trainer(batch_size, data_dir, dev_dataset, epochs, learning_rate, max_seq_length, model,
                       task_name, tokenizer, output_dir, train_dataset, warmup_steps, weight_decay):
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
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=dev_dataset,  # evaluation dataset
        compute_metrics=compute_metrics
    )
    # Train pre-trained model
    logger.info("***** Running training *****")
    trainer.train()
    logger.info("Training finished.")
    test_dataset = get_task_dataset(task_name, set_name='test', tokenizer=tokenizer,
                                    raw_data_dir=data_dir, max_seq_length=max_seq_length)
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions[1], axis=1)
    print(classification_report())


def compute_metrics(eval_pred):
    pred, labels = eval_pred
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)
    return {"accuracy": accuracy, "f1": f1}
