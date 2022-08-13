from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, Dataset
from .tokenizer import tokenizer

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

    
def get_data_loader(batch_size, config, split="train"):
    model_size = config.model_size
    dataset = load_dataset("papluca/language-identification")
    dataset = dataset[split]
    lang = {lang for lang in dataset["labels"]}
    lang = {l: i for i, l in enumerate(lang)}
    class_count = max(lang)
    dataset = dataset.map(lambda x: {"text": tokenizer(x['text'])}, batched=True)
    dataset = dataset.map(lambda x: partition(x, model_size), batched=True)
    dataset = dataset.map(lambda x: {"labels": lang[x["labels"]]})
    dataset = dataset.map(lambda x: {"text": pad(x["text"], model_size)})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = dataset.with_format("torch", device=device)
    print("BADRANX", len(dataset))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
