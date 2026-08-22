import torch
from torch import nn
import torch.nn.functional as functional

class LatentDynamicsModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.l1 = nn.Linear(2, config.dense_size // 4)
        self.l2 = nn.Linear(config.dense_size // 4, config.dense_size // 2)

        self.low_out = nn.Linear(
            config.dense_size // 2,
            config.dense_size
        )

        self.high_out = nn.Linear(
            config.dense_size // 2,
            config.dense_size
        )

    def forward(self, x, y):
        in_val = torch.cat([x, y], dim=-1)

        x = functional.gelu(self.l1(in_val))
        x = functional.gelu(self.l2(x))

        low = self.low_out(x)
        high = self.high_out(x)

        return low, high