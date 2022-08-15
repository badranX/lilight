from dataclasses import dataclass

@dataclass
class Config:
    model_size: int
    model_d: int
    vocab_size: int
    class_count: int
