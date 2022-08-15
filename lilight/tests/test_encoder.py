from .. import Encoder, config

import torch
from matplotlib import pyplot as plt

def test_encoder():
    e_d = 64
    e_size = 4
    vocab_size = 256
    language_count = 2
    config = config.Config(e_size, e_d, vocab_size, language_count)
    e = Encoder(config)
    txt = "absd"
    x = list(map(ord, txt))
    x = torch.tensor(x, dtype=torch.int)
    x = x.reshape(1, -1)

    val = e(x)
    #print(e.seq_embedding.embedding.weight.shape)
    #plt.imshow(p.detach()[0], cmap='gray')
    #plt.show()
    
    assert val != None
