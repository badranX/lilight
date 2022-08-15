
class Labels():
    def __init__(self, labels2idx):
        self._labels2idx = labels2idx
        self._idx2label = {v:k for k, v in labels2idx.items()}

    def idx2label(idxs):
        out = map(lambda x: self._idx2label[x], idxs)
        return list(out)


class LangSymbol:
    def __init__(self):
        self.lang2idx = {}
        self.max_id = 0
    def __call__(self, lang):
        return self.lang2idx.get(lang, self.max_id)

def split_or_pad(txt, max_size):
    if len(txt) < max_size:
        return 

def tokenizer(txts):
    txts = map(lambda x: list(bytearray(x.encode("utf-8"))), txts)
    print(txts)
    return list(txts)

def dokenizer(txts):
    tmp = lambda x: ''.join(x)
    txts = map(tmp, txts)
    return list(txts)

def pad(tokens_list, model_size):
    #TODO optemize
    fill_size = lambda x: len(x) % model_size
    max_size = model_size - len(tokens_list) 
    padder = lambda x: x + [0]*max(0, fill_size(x))
    tokens = map(padder, tokens_list)
    return padder(tokens)
