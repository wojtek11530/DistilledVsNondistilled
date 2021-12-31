import logging
import os
from typing import List, Optional, Union, Dict

import torch
import torch.utils.data
from transformers import BatchEncoding, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid: str, text_a: str, text_b: Optional[str] = None, label: Optional[str] = None):
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
    """Base class for data2 converters for sequence classification data2 sets."""

    def get_train_examples(self, data_dir: str):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir: str):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir: str):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_set_type_path(self, data_dir: str, set_type: str):
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data2 set."""
        raise NotImplementedError()

    @classmethod
    def _read_txt(cls, input_file: str) -> List[str]:
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='UTF-8') as f:
            lines = f.read().splitlines()
        return lines


class MultiemoProcessor(DataProcessor):
    """Processor for the Multiemo data2 set"""

    def __init__(self, lang: str, domain: str, kind: str):
        super(MultiemoProcessor, self).__init__()
        self.lang = lang.lower()
        self.domain = domain.lower()
        self.kind = kind.lower()

    def get_train_examples(self, data_dir: str) -> List[InputExample]:
        """See base class."""
        file_path = self.get_set_type_path(data_dir, 'train')
        logger.info(f"LOOKING AT {file_path}")
        return self._create_examples(self._read_txt(file_path), "train")

    def get_dev_examples(self, data_dir: str) -> List[InputExample]:
        """See base class."""
        file_path = self.get_set_type_path(data_dir, 'dev')
        return self._create_examples(self._read_txt(file_path), "dev")

    def get_test_examples(self, data_dir: str) -> List[InputExample]:
        """See base class."""
        file_path = self.get_set_type_path(data_dir, 'test')
        return self._create_examples(self._read_txt(file_path), "test")

    def get_set_type_path(self, data_dir: str, set_type: str) -> str:
        return os.path.join(data_dir, self.domain + '.' + self.kind + '.' + set_type + '.' + self.lang + '.txt')

    def get_labels(self) -> List[str]:
        """See base class."""
        if self.kind == 'text':
            return ["meta_amb", "meta_minus_m", "meta_plus_m", "meta_zero"]
        else:
            return ["z_amb", "z_minus_m", "z_plus_m", "z_zero"]

    @staticmethod
    def _create_examples(lines: List[str], set_type: str) -> List[InputExample]:
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            split_line = line.split('__label__')
            text_a = split_line[0]
            label = split_line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings: BatchEncoding, labels: Optional[torch.Tensor] = None):
        self.data = encodings.data
        self.n_examples = len(self.data['input_ids'])
        if labels is not None:
            self.data.update({'labels': labels})

    def __getitem__(self, idx: int):
        return {key: self.data[key][idx] for key in self.data.keys()}

    def __len__(self):
        return self.n_examples


class SmartCollator:
    def __init__(self, pad_token_id: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pad_token_id = pad_token_id

    def collate_batch(self, batch) -> Dict[str, torch.Tensor]:
        max_size = max([len(ex['input_ids']) for ex in batch])

        batch_inputs = list()
        batch_attention_masks = list()
        batch_token_type_ids = list()
        labels = list()

        for item in batch:
            batch_inputs += [pad_seq(item['input_ids'], max_size, self.pad_token_id)]
            batch_attention_masks += [pad_seq(item['attention_mask'], max_size, 0)]
            if 'labels' in item.keys():
                labels.append(item['labels'])
            if 'token_type_ids' in item.keys():
                batch_token_type_ids += [pad_seq(item['token_type_ids'], max_size, 0)]

        out_batch = {
            "input_ids": torch.tensor(batch_inputs, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_masks, dtype=torch.long)
        }
        if len(labels) != 0:
            out_batch.update({'labels': torch.tensor(labels)})
        if len(batch_token_type_ids) != 0:
            out_batch.update({'token_type_ids': torch.tensor(batch_token_type_ids, dtype=torch.long)})

        return out_batch


def pad_seq(seq: List[int], max_batch_len: int, pad_value: int) -> List[int]:
    return seq + (max_batch_len - len(seq)) * [pad_value]


def get_num_labels(task_name: str) -> int:
    processor = get_task_processor(task_name)
    label_list = processor.get_labels()
    num_labels = len(label_list)
    return num_labels


def get_labels(task_name: str) -> List[str]:
    processor = get_task_processor(task_name)
    return processor.get_labels()


def get_task_dataset(task_name: str, set_name: str, tokenizer: PreTrainedTokenizerBase,
                     raw_data_dir: str, max_seq_length: Optional[int] = None) -> Dataset:
    processor = get_task_processor(task_name)
    output_mode = get_output_mode(task_name)

    examples = get_examples_from_dataset(processor, raw_data_dir, set_name)

    label_list = processor.get_labels()
    label_map = {label: i for i, label in enumerate(label_list)}

    texts_a = []
    texts_b = []
    label_ids = []
    for (ex_index, example) in enumerate(examples):
        texts_a.append(example.text_a)
        if example.text_b is not None:
            texts_b.append(example.text_b)
        label_id = get_label_id(example, label_map, output_mode)
        label_ids.append(label_id)

    if len(texts_a) == len(texts_b):
        text_tokenized = tokenizer(
            texts_a,
            texts_b,
            add_special_tokens=True,
            truncation=True,
            padding=False,
            max_length=max_seq_length,
            return_attention_mask=True,
            return_token_type_ids=True
        )
    else:
        text_tokenized = tokenizer(
            texts_a,
            add_special_tokens=True,
            truncation=True,
            padding=False,
            max_length=max_seq_length,
            return_attention_mask=True,
            return_token_type_ids=False
        )

    if output_mode == "classification":
        all_label_ids = torch.tensor(label_ids, dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor(label_ids, dtype=torch.float)
    else:
        raise ValueError('Incorrect output mode')

    dataset = Dataset(text_tokenized, all_label_ids)
    return dataset


def get_task_dataset_dir(task_name: str, set_name: str, raw_data_dir: str) -> str:
    processor = get_task_processor(task_name)
    return processor.get_set_type_path(raw_data_dir, set_name)


def get_examples_from_dataset(processor, raw_data_dir, set_name):
    if set_name.lower() == 'train':
        examples = processor.get_train_examples(raw_data_dir)
    elif set_name.lower() == 'dev':
        examples = processor.get_dev_examples(raw_data_dir)
    elif set_name.lower() == 'test':
        examples = processor.get_test_examples(raw_data_dir)
    else:
        raise ValueError(
            '{} as set name not available for now, use \'train\', \'dev\' or \'test\' instead'.format(set_name))
    return examples


def get_output_mode(task_name: str) -> str:
    if 'multiemo' in task_name:
        output_mode = output_modes['multiemo']
    else:
        output_mode = output_modes[task_name]
    return output_mode


def get_task_processor(task_name: str) -> DataProcessor:
    if 'multiemo' in task_name:
        _, lang, domain, kind = task_name.split('_')
        processor = MultiemoProcessor(lang, domain, kind)
    else:
        if task_name not in processors:
            raise ValueError("Task not found: %s" % task_name)
        else:
            processor = processors[task_name]()
    return processor


def get_label_id(example: InputExample, label_map: dict, output_mode: str) -> Union[int, float]:
    if output_mode == "classification":
        label_id = label_map[example.label]
    elif output_mode == "regression":
        label_id = float(example.label)
    else:
        raise KeyError(output_mode)
    return label_id


processors = {
    "multiemo": MultiemoProcessor,
}

output_modes = {
    "multiemo": "classification",
}
