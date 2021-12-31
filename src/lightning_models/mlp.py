from typing import Any, Dict, Tuple

import torch
import torchmetrics
from pytorch_lightning import LightningModule
from torch.nn import functional as F


class MLPClassifier(LightningModule):
    def __init__(self, input_size: int = 100, output_size: int = 4, hidden_size: int = 24,
                 learning_rate: float = 1e-3, weight_decay: float = 1e-5, dropout: float = 0.5):
        super(MLPClassifier, self).__init__()
        self.save_hyperparameters()

        self._layer_1 = torch.nn.Linear(input_size, hidden_size)
        self._layer_2 = torch.nn.Linear(hidden_size, output_size)
        self._dropout = torch.nn.Dropout(p=dropout)

        self._learning_rate = learning_rate
        self._weight_decay = weight_decay

        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = self._layer_1(x)
        x = F.relu(x)
        x = self._dropout(x)
        x = self._layer_2(x)
        return x

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), self._learning_rate, weight_decay=self._weight_decay)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) \
            -> Dict[str, Any]:
        loss, y_true, probs = self._shared_step(batch)
        self.train_accuracy(probs, y_true)
        self.log('train_acc', self.train_accuracy, on_step=False, on_epoch=True)
        self.log('train_loss', loss.item(), on_step=True, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, val_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) \
            -> Dict[str, Any]:
        loss, y_true, probs = self._shared_step(val_batch)
        self.val_accuracy(probs, y_true)
        self.log('val_acc', self.val_accuracy, on_epoch=True)
        self.log('val_loss', loss.item(), on_epoch=True, prog_bar=True)
        return {'val_loss': loss}

    def _shared_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y_true = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y_true)
        probs = torch.softmax(logits, dim=1)
        return loss, y_true, probs
