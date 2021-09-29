import logging
import os
from typing import Optional

import torch
from torch.utils.data import TensorDataset, DataLoader

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_txt(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            text = f.read()


class MultiemoProcessor(DataProcessor):
    """Processor for the Multiemo data set"""

    def __init__(self, lang: str, domain: str, kind: str):
        super(MultiemoProcessor, self).__init__()
        self.lang = lang.lower()
        self.domain = domain.lower()
        self.kind = kind.lower()

    def get_train_examples(self, data_dir):
        """See base class."""
        file_path = os.path.join(data_dir, self.domain + '.' + self.kind + '.train.' + self.lang + '.txt')
        logger.info(f"LOOKING AT {file_path}")
        return self._create_examples(self._read_txt(file_path), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        file_path = os.path.join(data_dir, self.domain + '.' + self.kind + '.dev.' + self.lang + '.txt')
        return self._create_examples(self._read_txt(file_path), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        file_path = os.path.join(data_dir, self.domain + '.' + self.kind + '.test.' + self.lang + '.txt')
        return self._create_examples(self._read_txt(file_path), "test")

    def get_labels(self):
        """See base class."""
        if self.kind == 'text':
            return ["__label__meta_amb", "__label__meta_minus_m", "__label__meta_plus_m", "___label__meta_zero"]
        else:
            return ["__label__z_amb", "__label__z_minus_m", "__label__z_plus_m", "___label__z_zero"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            split_line = line.split('__label__')
            text_a = split_line[0]
            label = split_line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings: torch.Tensor, labels: Optional[torch.Tensor] = None):
        self.n_examples = len(labels)
        self.inputs = encodings
        self.sequence_len = self.inputs['input_ids'].shape[-1]
        self.inputs.update({'labels': torch.tensor(labels)})

    def __getitem__(self, idx):
        return {key: self.inputs[key][idx] for key in self.inputs.keys()}

    def __len__(self):
        return self.n_examples


processors = {
    "multiemo": MultiemoProcessor,
}

output_modes = {
    "multiemo": "classification",
}


def get_task_dataloader(task_name: str, set_name: str, tokenizer, raw_data_dir, batch_size, max_seq_length):
    if 'multiemo' in task_name:
        _, lang, domain, kind = task_name.split('_')
        processor = MultiemoProcessor(lang, domain, kind)
    else:
        if task_name not in processors:
            raise ValueError("Task not found: %s" % task_name)
        else:
            processor = processors[task_name]()

    output_mode = output_modes[task_name]

    label_list = processor.get_labels()
    if set_name.lower() == 'train':
        examples = processor.get_train_examples(raw_data_dir)
    elif set_name.lower() == 'dev':
        examples = processor.get_dev_examples(raw_data_dir)
    elif set_name.lower() == 'test':
        examples = processor.get_test_examples(raw_data_dir)
    else:
        raise ValueError(
            '{} as set name not available for now, use \'train\', \'dev\' or \'test\' instead'.format(set_name))

    label_map = {label: i for i, label in enumerate(label_list)}
    texts_a = []
    texts_b = []
    label_ids = []
    for (ex_index, example) in enumerate(examples):
        texts_a.append(example.text_a)

        if example.text_b is not None:
            texts_b.append(example.text_b)

        label_id = _get_label_id(example, label_map, output_mode)
        label_ids.append(label_id)

    if len(texts_a) == len(texts_b):
        text_tokenized = tokenizer(texts_a, texts_b, truncation=True, padding=True, return_tensors='pt',
                                   max_length=max_seq_length)
    else:
        text_tokenized = tokenizer(texts_a, truncation=True, padding=True, return_tensors='pt',
                                   max_length=max_seq_length)

    if output_mode == "classification":
        all_label_ids = torch.tensor(label_ids, dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor(label_ids, dtype=torch.float)

    dataset = Dataset(text_tokenized, all_label_ids)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True if set_name != 'test' else False)
    return examples, dataloader, all_label_ids


def _get_label_id(example, label_map, output_mode):
    if output_mode == "classification":
        label_id = label_map[example.label]
    elif output_mode == "regression":
        label_id = float(example.label)
    else:
        raise KeyError(output_mode)
    return label_id
