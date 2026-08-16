import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F


def wavelet_decompose(x):
    B, T, N = x.shape

    x = x.permute(0, 2, 1)
    x = x.reshape(B, N, T // 2, 2)

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
        self.query = nn.Linear(config.dense_size, config.dense_size // config.n_heads)
        self.keys = nn.Linear(config.dense_size, config.dense_size // config.n_heads)
        self.values = nn.Linear(config.dense_size, config.dense_size // config.n_heads)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, low):
        B, T, N, D = low.shape
        low = low.permute(0, 2, 1, 3) # B, N, T, D
        q, k, v = self.query(low), self.keys(low), self.values(low)
        pre_softmax = q @ k.transpose(-1, -2) # (T, D) @ (D, T) = (T, T)
        scores = F.softmax(pre_softmax, dim=-1)
        return self.dropout(scores @ v) #(T, T) @ (T, D) = (T, D)

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.heads = nn.ModuleList([Head(config) for _ in range(config.n_heads)])
        self.projection = nn.Linear(config.dense_size, config.dense_size)

    def forward(self, low):
        output = torch.cat([head(low) for head in self.heads], dim=-1)
        output = self.projection(output)
        return output.permute(0, 2, 1, 3) #(B, N, T, D) ----> (B, T, N, D)

class DialatedCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.conv1 = nn.Conv1d(config.dense_size, config.dense_size, kernel_size=3, dilation=1)
        self.conv2 = nn.Conv1d(config.dense_size, config.dense_size, kernel_size=3, dilation=1)
    
    def forward(self, high):
        B, T, N, D = high.shape
        high = high.permute(0, 2, 3, 1).reshape(B * N, D, T)

        high = F.pad(high, (2, 0))
        x = F.relu(self.conv1(high))
        x = F.pad(x, (2, 0))
        x = F.relu(self.conv2(x))

        T_out = x.shape[-1]
        out = x.reshape(B, N, D, T_out).permute(0, 3, 1, 2) #(B, T, N, D)
        return out

class DualFrequencySpatiotemporalEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feature_proj = nn.Linear(config.n_features, config.dense_size)
        self.temporal_attention = MultiHeadAttention(config)
        self.dialated_cnn = DialatedCNN(config)

        self.cross_stock_low = MultiHeadAttention(config)
        self.cross_stock_high = MultiHeadAttention(config)

    def forward(self, low, high):
        low = self.feature_proj(low)
        high = self.feature_proj(high)

        low_out = self.temporal_attention(low)
        high_out = self.dialated_cnn(high)

        B, T, N, D = low_out.shape
        B2, T2, N2, D2 = high_out.shape

        if B != B2 and T != T2 and N != N2 and D != D2:
            raise ValueError("There is a missmatch in the low_out shape and high_out shape")

        low_prev = low_out.permute(0, 3, 2, 1)
        high_prev = high_out.permute(0, 3, 2, 1)

        low_out = self.cross_stock_low(low_prev)
        high_out = self.cross_stock_low(high_prev)
        
        return low_out, high_out


class TransformerPred(nn.Module):
    def __init__(self, config):
        super().__init__()