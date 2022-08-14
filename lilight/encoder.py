import torch
from torch import nn
from torch.functional import F
from dataclasses import dataclass

@dataclass
class EncoderConfig:
    model_size: int
    model_d: int
    vocab_size: int
    class_count: int


class SequenceEmbedding(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        I = torch.eye(config.model_size).unsqueeze(0)
        self.register_buffer("I", torch.eye(config.model_size))
        self.keys = nn.Embedding(config.vocab_size, config.model_d )
        self.vals = nn.Embedding(config.vocab_size, config.model_d)
        self.fuse = nn.Linear(config.model_d + config.model_size, config.model_d)

    def forward(self, x):
        keys = self.keys(x)
        vals = self.vals(x)
        pos = self.I.expand(x.shape[0], *[-1]*len(self.I.shape))
        keys = torch.cat([keys, pos], dim=-1)
        vals = torch.cat([vals, pos], dim=-1)
        vals = self.fuse(vals)
        keys = self.fuse(keys)
        return keys, vals


class Encoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.model_d = config.model_d
        self.model_size = config.model_size
        self.class_count = config.class_count

        self.seq_embedding = SequenceEmbedding(config)
        self.mlp = nn.Sequential(
            nn.Linear(self.model_d, 1),
            nn.Sigmoid())

        self.classifier_mlp = nn.Sequential(
                nn.Linear(self.model_size, self.class_count),
                nn.Sigmoid())

        self.test = nn.Sequential(
                nn.Linear(self.model_d*self.model_size, self.class_count),
                nn.Sigmoid())

    def forward(self, x):
        keys, vals = self.seq_embedding(x)
        #test
        #keys = keys[:,0,:]
        #done test
        scores = keys @ keys.transpose(-2, -1)
        probs = torch.softmax(scores, -1)
        attention = probs @ keys 
        #test
        x = self.test(attention.view(attention.shape[0], -1))
        x = F.softmax(x, dim=-1)
        return x
        x = self.mlp(attention).squeeze(-1)
        x = self.classifier_mlp(x)
        x = F.softmax(x, dim=-1)
        return x
