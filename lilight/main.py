from .encoder import Encoder
from .config import *
from .dataset import Dataset
from .train import *
import argparse


def start_train():
    config = Config()
    traindata = Dataset(config)
    train_dataloader = traindata.get_dataloader(batch_size = config.class_count, split="train")
    eval_dataloader = traindata.get_dataloader(batch_size = config.class_count, split="validation")
    config.train_dataloader = train_dataloader
    config.eval_dataloader = train_dataloader
    model = Encoder(config)
    train(model, config)


if __name__ == "__main__":
    start_train()
