from .encoder import Encoder, EncoderConfig
from .dataset import get_data_loader
from .train import *
import argparse


def start_train():
    class_count = 210
    config = EncoderConfig(model_size=64, model_d=64, vocab_size=256, class_count=class_count)
    dataloader = get_data_loader(batch_size = 2, config=config, split="train")
    print(len(dataloader))
    model = Encoder(config)
    train(model, dataloader, epochs=10)


if __name__ == "__main__":
    start_train()
