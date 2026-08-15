import torch
from torch import nn
import torch.optim as optim

def wavelet_decompose(x):
    return x

class Decoupling(nn.Module):
    def __init__(self, config):
        super().__init__()
        pass
    def forward(self, x):
        B, T, N, F = x.shape

        returns = x[:, :, :, 0]
        other_features = x[:, :, :, 1:]
        low_returns, high_returns = wavelet_decompose(returns)

        x_low = torch.cat(
            [low_returns,
            high_returns],
            dim=-1
        )

class TransformerPred(nn.Module):
    def __init__(self, config):
        super().__init__()
