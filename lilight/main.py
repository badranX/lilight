from .encoder import Encoder
from .config import *
from .dataset import TrainData
from .train import *
import argparse


def start_train():
    config = Config(model_size=128, model_d=64, vocab_size=256, class_count=None)
    traindata = TrainData(config)
    dataloader = traindata.get_dataloader(batch_size = 64, split="train")
    print(len(dataloader))
    model = Encoder(config)
    train(model, dataloader, lr=0.0025, epochs=10)


if __name__ == "__main__":
    start_train()
