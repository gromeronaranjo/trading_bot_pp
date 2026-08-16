import torch
from torch import nn
import torch.optim as optim


def wavelet_decompose(x):
    B, T, N = x.shape

    x = x.permute(0, 2, 1)
    x = x.reshape(B, N, T/ 2, 2)

    average = x.mean(dim=-1)
    difference = (x[..., 0] - x[..., 1]) / 2

    low = torch.stack([average, average], dim=-1)
    high = torch.stack([difference, -difference], dim=-1)

    low = low.reshape(B, N, T).permute(0, 2, 1)
    high = high.reshape(B, N, T).permute(0, 2, 1)

    return low, high


class Decoupling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        returns = x[..., 0]
        other_features = x[..., 1:]

        low_returns, high_returns = wavelet_decompose(returns)

        low_feature_map = torch.cat([low_returns.unsqueeze(-1), other_features],dim=-1)
        high_feature_map = torch.cat([high_returns.unsqueeze(-1), other_features], dim=-1)

        return low_feature_map, high_feature_map


class TransformerPred(nn.Module):
    def __init__(self, config):
        super().__init__()