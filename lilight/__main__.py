import os
import torch
from collections import Counter
import argparse
from .tokenizer import tokenizer
from .dataset import partition, pad

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='File language identification')
    parser.add_argument('file_path', type=str,
                        help='A document file written in one of the supported languages')

    args = parser.parse_args()

    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'model.pt')
    model = torch.load(filename)
    config = model.config

    print("The supported languages are:")
    print(config.labels.idx2label.values())
    print("check hugginface dataseet 'papluca/language-identification'",
            "for language codes")
    print("---------------------------")
    print()

    with open(args.file_path, 'r') as f:
        lines = [line.rstrip() for line in f]
        one_line = ' '.join(lines)

        MAX_CHARS = 1000
        one_line = one_line[0:MAX_CHARS]

        one_line = tokenizer([one_line])
        one_line = one_line[0]

        s = config.model_size
        tmp = set(range(0, len(one_line) - s)).union([0])
        text_rows = [one_line[i: i + s] for i in tmp]
        #print(bytearray(text_rows[0]).decode('utf-8'))

        unknown_labels = ['_']*len(text_rows)
        rows = {'text': text_rows, 'labels': unknown_labels}
        rows = partition(rows, config.model_size)
        rows['text'] = map(lambda x: pad(x, config.model_size), rows["text"])
        rows['text'] = list(rows['text'])

        x = rows["text"]
        x = torch.tensor(x)
        y = model(x)
        y = torch.argmax(y, dim=-1)
        y = map(lambda x: x.item(), y)
        y = map(lambda idx: config.labels.idx2label[idx], y)
        y = Counter(y)
        max_lang = max(y.keys(), key= lambda idx: y[idx])
        print("the language is: ", max_lang)
        print("the total guesses are: ")
        print(y)
