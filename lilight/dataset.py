from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, Dataset
from .tokenizer import tokenizer, Labels



def partition(rows, model_size):
    tmp = lambda x: [x[i: i + model_size] for i in range(0, len(x), model_size)]
    result = [(tmp(txt), label) for txt, label in zip(rows["text"], rows["labels"])]
    result = [(txts, len(txts)*[label]) for txts, label in result]
    result_txt = [txt for txts, _ in result for txt in txts]
    result_label = [label for _, labels in result for label in labels]
    #print(result_label)
    return {"text": result_txt, "labels": result_label}


def pad(txt, max_size):
    if len(txt) < max_size:
        txt += [0]*(max_size - len(txt))
    return txt

class TrainData:

    def __init__(self, config):
        model_size = config.model_size
        datasets = load_dataset("papluca/language-identification")
    
        lang = {lang for lang in datasets["train"]["labels"]}
        lang = {l: i for i, l in enumerate(lang)}
        config.class_count = len(lang)
        self.labels = Labels(lang)

        new_ds = {}
        for key, dataset in datasets.items():
            dataset = dataset.map(lambda x: {"text": tokenizer(x['text'])}, batched=True)
            dataset = dataset.map(lambda x: partition(x, model_size), batched=True)
            dataset = dataset.map(lambda x: {"labels": lang[x["labels"]]})
            dataset = dataset.map(lambda x: {"text": pad(x["text"], model_size)})
            new_ds[key] = dataset

        self.datasets = new_ds

    def get_dataloader(self, batch_size, split="train"):
        dataset = self.datasets[split]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dataset = dataset.with_format("torch", device=device)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return self.dataloader
