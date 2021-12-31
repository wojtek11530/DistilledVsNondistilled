import os
from typing import List, Optional, Tuple

import numpy as np
from pytorch_lightning import LightningDataModule
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch

from src.data.data_processing import get_examples_from_dataset, get_task_processor, get_output_mode, get_label_id
from src.settings import MODELS_FOLDER_2


class LabseDataModule(LightningDataModule):
    def __init__(self, task_name: str, raw_data_dir: str, batch_size: int = 32):
        super().__init__()
        self.task_name = task_name
        self.data_dir = raw_data_dir
        self.batch_size = batch_size

        self.model = SentenceTransformer("sentence-transformers/LaBSE",
                                         cache_folder=os.path.join(MODELS_FOLDER_2, 'LaBSE'))

        self.train = None
        self.dev = None
        self.test = None
        self.setup()

    def setup(self, stage: Optional[str] = None):
        self.train = LabseDataset(
            task_name=self.task_name, raw_data_dir=self.data_dir, set_name='train', embedder=self.model
        )
        self.dev = LabseDataset(
            task_name=self.task_name, raw_data_dir=self.data_dir, set_name='dev', embedder=self.model
        )
        self.test = LabseDataset(
            task_name=self.task_name, raw_data_dir=self.data_dir, set_name='test', embedder=self.model
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dev,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
        )


class LabseDataset(Dataset):
    def __init__(self, task_name: str, raw_data_dir: str, set_name: str, embedder: SentenceTransformer):
        super().__init__()
        processor = get_task_processor(task_name)
        output_mode = get_output_mode(task_name)
        examples = get_examples_from_dataset(processor, raw_data_dir, set_name)

        label_list = processor.get_labels()
        label_map = {label: i for i, label in enumerate(label_list)}

        texts = []
        label_ids = []
        for (ex_index, example) in enumerate(examples):
            texts.append(example.text_a)
            label_id = get_label_id(example, label_map, output_mode)
            label_ids.append(label_id)

        if output_mode == "classification":
            all_label_ids = torch.tensor(label_ids, dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor(label_ids, dtype=torch.float)
        else:
            raise ValueError('Incorrect output mode')

        desc = f"Getting embeddings for {task_name}-{set_name}"
        self.embedding_data = [np.array(embedder(text))for text in tqdm(texts, desc=desc)]
        self.labels = all_label_ids

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        return self.embedding_data[index], self.labels[index]

    def __len__(self) -> int:
        return len(self.labels)
