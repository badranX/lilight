from dataclasses import dataclass
from torch.utils.data import DataLoader
from .tokenizer import Labels
from .dataset import Dataset

@dataclass
class HyperParameters:
    model_size: int = 64
    model_d: int = 64
    vocab_size: int = 256
    batch_size: int = 64
    lr: float = 0.0005 
    epochs: int = 30


@dataclass
class Config(HyperParameters):
    class_count: int = None
    train_dataloader: DataLoader = None
    eval_dataloader: DataLoader = None
    labels: Labels = None
