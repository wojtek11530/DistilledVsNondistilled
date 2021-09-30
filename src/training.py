import os

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, TrainingArguments, AutoTokenizer, Trainer

from src.data_processing import get_task_dataset
from settings import MODELS_FOLDER


def train_model(model_name: str, task_name: str, data_dir: str, batch_size: int = 32, max_seq_length: int = 512):
    num_labels = 2
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=os.path.join(MODELS_FOLDER, model_name))

    train_dataset = get_task_dataset(task_name, set_name='train', tokenizer=tokenizer,
                                     raw_data_dir=data_dir, max_seq_length=max_seq_length)
    dev_dataset = get_task_dataset(task_name, set_name='dev', tokenizer=tokenizer,
                                   raw_data_dir=data_dir, max_seq_length=max_seq_length)

    # Define Trainer
    training_args = TrainingArguments(
        output_dir=os.path.join(MODELS_FOLDER, model_name, task_name),
        num_train_epochs=3,
        warmup_steps=0,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        per_device_train_batch_size=batch_size,  # batch size per device during training
        per_device_eval_batch_size=batch_size,  # batch size for evaluation
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,

    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=dev_dataset,  # evaluation dataset
        compute_metrics=compute_metrics
    )

    # Train pre-trained model
    trainer.train()

    test_dataset = get_task_dataset(task_name, set_name='test', tokenizer=tokenizer,
                                    raw_data_dir=data_dir, max_seq_length=max_seq_length)

    predictions = trainer.predict(test_dataset)


def compute_metrics(eval_pred):
    pred, labels = eval_pred
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "f1": f1}
