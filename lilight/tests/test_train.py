from ..encoder import Encoder, EncoderConfig
from torch.utils.data import DataLoader, Dataset
from ..dataset import get_data_loader
from ..train import *
import argparse

from datasets import Dataset

def test_train():
    class_count = 210
    config = EncoderConfig(model_size=100, model_d=4, vocab_size=256, class_count=class_count)
    ds = Dataset.from_dict({"text": [torch.randint(0,40, [100]) for i in range(40)], "labels": [i for i in range(40)]})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = ds.with_format('torch', device=device)

    dataloader = DataLoader(ds, batch_size=2)
    model = Encoder(config)
    train(model, dataloader, epochs=100)


if __name__ == "__main__":
    start_train()
