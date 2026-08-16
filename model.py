import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F


def wavelet_decompose(x):
    B, T, N = x.shape

    x = x.permute(0, 2, 1)
    x = x.reshape(B, N, T/2, 2)

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

class Head(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.query = nn.Linear(config.dense_size, config.dense_size/config.n_heads)
        self.keys = nn.Linear(config.dense_size, config.dense_size/config.n_heads)
        self.values = nn.Linear(config.dense_size, config.dense_size/config.n_heads)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, low):
        B, T, N, D = low.shape
        low = low.permute(-1, 1) # B, N, T, D
        q, k, v = self.query(low), self.keys(low), self.values(low)
        pre_softmax = q @ k.permute(-1, -2) # (T, D) @ (D, T) = (T, T)
        scores = F.softmax(pre_softmax)
        return self.dropout(scores @ v) #(T, T) @ (T, D) = (T, D)

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.projection = nn.Linear(config.dense_size, config.dense_size)

    def forward(self, low):
        output = torch.cat([Head(low) for _ in range(self.config.n_heads)], dim=-1)
        return self.projection(output)

class DualFrequencySpatiotemporalEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feature_proj = nn.Linear(config.n_features, config.dense_size)

    def forward(self, low, high):
        low = self.feature_proj(low)
        high = self.feature_proj(high)

        temporal_attention_low = MultiHeadAttention(low)

class TransformerPred(nn.Module):
    def __init__(self, config):
        super().__init__()