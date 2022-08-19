import torch
from torch import nn
from torch.functional import F
from dataclasses import dataclass
from .config import Config

class SequenceEmbedding(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        I = torch.eye(config.model_size).unsqueeze(0)
        self.register_buffer("I", torch.eye(config.model_size))
        self.keys = nn.Embedding(config.vocab_size, config.model_d )
        self.vals = nn.Embedding(config.vocab_size, config.model_d)
        self.fuse = nn.Sequential(
                nn.Linear(config.model_d + config.model_size, config.model_d),
                nn.Sigmoid())

    def forward(self, x):
        mask = x != 0
        mask = mask.unsqueeze(-1).detach()
        keys = self.keys(x)
        vals = self.vals(x)
        pos = self.I.expand(x.shape[0], *[-1]*len(self.I.shape))
        keys = torch.cat([keys, pos], dim=-1)
        vals = torch.cat([vals, pos], dim=-1)
        vals = self.fuse(vals)*mask
        keys = self.fuse(keys)*mask
        return keys, vals


class Encoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.vocab_size = config.vocab_size
        self.model_d = config.model_d
        self.model_size = config.model_size
        self.class_count = config.class_count

        self.seq_embedding = SequenceEmbedding(config)
        self.mlp = nn.Sequential(
            nn.Linear(self.model_d, self.model_d//2),
            nn.ReLU(),
            nn.Linear(self.model_d//2, 1),
            nn.Sigmoid()
            )

        self.classifier_mlp = nn.Sequential(
                nn.Linear(self.model_size, self.model_size),
                nn.ReLU(),
                nn.Linear(self.model_size, self.class_count)
                )
                #nn.Sigmoid())

    def forward(self, x):
        keys, vals = self.seq_embedding(x)
        scores = keys @ vals.transpose(-2, -1)
        #probs = torch.sigmoid(scores)
        probs = torch.softmax(scores, -1)
        attention = probs @ vals
        x = self.mlp(attention).squeeze(-1)
        x = self.classifier_mlp(x)
        return x
