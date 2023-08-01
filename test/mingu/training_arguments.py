from dataclasses import dataclass, field, fields
from typing import Callable, Optional, Union

import torch
from dataset import TimeSeriesValidSplit
from skorch.dataset import Dataset


@dataclass
class TrainingArguments:
    criterion: torch.nn = field(default=torch.nn.MSELoss)
    optimizer: torch.optim = field(default=torch.optim.AdamW)
    lr: float = field(default=0.01)
    max_epochs: int = field(default=10)
    batch_size: int = field(default=128)
    dataset: Optional[Dataset] = field(default=Dataset)
    callbacks: Optional[list[Callable]] = field(default=None)
    train_split: Optional[Union[int, float]] = field(default=0.75)
    device: str = field(default='cpu')

    def __post_init__(
        self,
    ):
        self.train_split = TimeSeriesValidSplit(train_size=self.train_split)
