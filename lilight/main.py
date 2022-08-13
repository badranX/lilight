from .encoder import Encoder, EncoderConfig
from .dataset import get_data_loader
from .train import *
import argparse


def start_train():
    class_count = 210
    config = EncoderConfig(model_size=54, model_d=12, vocab_size=256, class_count=class_count)
    dataloader = get_data_loader(batch_size = 10, config=config, split="train")
    model = Encoder(config)
    train(model, dataloader, epochs=2)


if __name__ == "__main__":
    start_train()
