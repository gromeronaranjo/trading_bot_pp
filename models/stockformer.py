import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as functional


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

        low_feature_map = torch.cat([low_returns.unsqueeze(-1), other_features], dim=-1)
        high_feature_map = torch.cat([high_returns.unsqueeze(-1), other_features], dim=-1)

        return low_feature_map, high_feature_map


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.l1 = nn.Linear(config.dense_size, config.dense_size)
        self.l2 = nn.Linear(config.dense_size, config.dense_size)
        self.norm = nn.LayerNorm(config.dense_size, elementwise_affine=False)

    def forward(self, x):
        residual = x
        x = functional.relu(self.l1(x))
        x = self.l2(x)
        x = x + residual
        return self.norm(x)


class Head(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.query = nn.Linear(config.dense_size, config.dense_size // config.n_heads)
        self.keys = nn.Linear(config.dense_size, config.dense_size // config.n_heads)
        self.values = nn.Linear(config.dense_size, config.dense_size // config.n_heads)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, low):
        B, T, N, D = low.shape
        low = low.permute(0, 2, 1, 3) # B, N, T, D
        q, k, v = self.query(low), self.keys(low), self.values(low)

        pre_softmax = q @ k.transpose(-1, -2) # (T, D) @ (D, T) = (T, T)
        pre_softmax = pre_softmax / ((D // self.config.n_heads) ** 0.5)

        mask = torch.tril(torch.ones(T, T, device=low.device)).bool()
        pre_softmax = pre_softmax.masked_fill(~mask, float("-inf"))

        scores = functional.softmax(pre_softmax, dim=-1)
        return self.dropout(scores @ v) #(T, T) @ (T, D) = (T, D)


class MultiHeadAttention(nn.Module):
    def __init__(self, config, head):
        super().__init__()
        self.config = config
        self.heads = nn.ModuleList([head(config) for _ in range(config.n_heads)])
        self.projection = nn.Linear(config.dense_size, config.dense_size)
        self.norm = nn.LayerNorm(config.dense_size, elementwise_affine=False)
        self.mlp = MLP(config)

    def forward(self, *x):
        residual = x[0]

        output = torch.cat([head(*x) for head in self.heads], dim=-1)
        output = self.projection(output)
        output = output.permute(0, 2, 1, 3) #(B, N, T, D) ----> (B, T, N, D)

        output = self.norm(output + residual)
        output = self.mlp(output)

        return output


class CrossStockHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.query = nn.Linear(config.dense_size, config.dense_size // config.n_heads)
        self.keys = nn.Linear(config.dense_size, config.dense_size // config.n_heads)
        self.values = nn.Linear(config.dense_size, config.dense_size // config.n_heads)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        q, k, v = self.query(x), self.keys(x), self.values(x)

        pre_softmax = q @ k.transpose(-1, -2)
        pre_softmax = pre_softmax / ((self.config.dense_size // self.config.n_heads) ** 0.5)

        scores = functional.softmax(pre_softmax, dim=-1)
        return self.dropout(scores @ v)


class CrossStockAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.heads = nn.ModuleList([CrossStockHead(config) for _ in range(config.n_heads)])
        self.projection = nn.Linear(config.dense_size, config.dense_size)
        self.norm = nn.LayerNorm(config.dense_size, elementwise_affine=False)
        self.mlp = MLP(config)

    def forward(self, x):
        residual = x

        output = torch.cat([head(x) for head in self.heads], dim=-1)
        output = self.projection(output)

        output = self.norm(output + residual)
        output = self.mlp(output)

        return output

class DialatedCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.conv1 = nn.Conv1d(config.dense_size, config.dense_size, kernel_size=2, dilation=1)
        self.conv2 = nn.Conv1d(config.dense_size, config.dense_size, kernel_size=2, dilation=2)

    def forward(self, high):
        B, T, N, D = high.shape
        high = high.permute(0, 2, 3, 1).reshape(B * N, D, T)

        high = functional.pad(high, (1, 0))
        x = functional.relu(self.conv1(high))
        x = functional.pad(x, (2, 0))
        x = functional.relu(self.conv2(x))

        T_out = x.shape[-1]
        out = x.reshape(B, N, D, T_out).permute(0, 3, 1, 2) #(B, T, N, D)
        return out


class FuturePredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictor = nn.Linear(config.time_steps, 2)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.predictor(x)
        return x.permute(0, 3, 1, 2)

class PearsonModule(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, x):
        B, T, N, F = x.shape()
        x = x.permute(0, 3, 1, 2)
        x = x[:, :, :, 0].squeeze(-1) #(B, T, N)
        coefs = torch.corrcoef(x)
        return coefs

class DualFrequencySpatiotemporalEncoder(nn.Module):
    def __init__(self, config, stock_graph_embedding):
        super().__init__()
        self.feature_proj = nn.Linear(config.n_features, config.dense_size)
        self.temporal_attention = MultiHeadAttention(config, Head)
        self.dialated_cnn = DialatedCNN(config)

        self.cross_stock_low = CrossStockAttention(config)
        self.cross_stock_high = CrossStockAttention(config)

        self.register_buffer(
            "stock_embedding",
            stock_graph_embedding.unsqueeze(0).unsqueeze(0)
        )

        self.time_embedding = nn.Parameter(
            torch.randn(1, config.time_steps, 1, config.dense_size)
        )

    def forward(self, low, high):
        low = self.feature_proj(low)
        high = self.feature_proj(high)

        low_out = self.temporal_attention(low)
        high_out = self.dialated_cnn(high)

        B, T, N, D = low_out.shape
        B2, T2, N2, D2 = high_out.shape

        if B != B2 or T != T2 or N != N2 or D != D2:
            raise ValueError("There is a mismatch in the low_out shape and high_out shape")

        low_out = low_out + self.stock_embedding + self.time_embedding
        high_out = high_out + self.stock_embedding + self.time_embedding

        low_out = self.cross_stock_low(low_out)
        high_out = self.cross_stock_high(high_out)

        return low_out, high_out

class CrossLowHighAttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.query = nn.Linear(config.dense_size, config.dense_size // config.n_heads)
        self.keys = nn.Linear(config.dense_size, config.dense_size // config.n_heads)
        self.values = nn.Linear(config.dense_size, config.dense_size // config.n_heads)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, low, high):
        B, T, N, D = low.shape
        low = low.permute(0, 2, 1, 3) # B, N, T, D
        high = high.permute(0, 2, 1, 3)

        q, k, v = self.query(low), self.keys(high), self.values(high)

        pre_softmax = q @ k.transpose(-1, -2) # (T, D) @ (D, T) = (T, T)
        pre_softmax = pre_softmax / ((D // self.config.n_heads) ** 0.5)

        mask = torch.tril(torch.ones(T, T, device=low.device)).bool()
        pre_softmax = pre_softmax.masked_fill(~mask, float("-inf"))

        scores = functional.softmax(pre_softmax, dim=-1)
        return self.dropout(scores @ v) #(T, T) @ (T, D) = (T, D)


class DualFrequencyFusionModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.low_attention = MultiHeadAttention(config, Head)
        self.cross_attention = MultiHeadAttention(config, CrossLowHighAttentionHead)

    def forward(self, low, high):
        low_out = self.low_attention(low)
        cross_out = self.cross_attention(low, high)

        return low_out + cross_out


class StockFormer(nn.Module):
    def __init__(self, config, stock_graph_embedding):
        super().__init__()
        self.config = config
        self.decoupling = Decoupling()

        self.duel_freq_spatio_temporal_encoder = DualFrequencySpatiotemporalEncoder(
            config,
            stock_graph_embedding
        )

        self.future_pred_low = FuturePredictor(config)
        self.future_pred_high = FuturePredictor(config)
        self.dual_frequency_fusion = DualFrequencyFusionModule(config)

        self.return_proj = nn.Linear(config.dense_size, 1)
        self.direction_proj = nn.Linear(config.dense_size, 1)

    def forward(self, x):
        low, high = self.decoupling(x)
        low, high = self.duel_freq_spatio_temporal_encoder(low, high)
        low, high = self.future_pred_low(low), self.future_pred_high(high)
        out = self.dual_frequency_fusion(low, high)

        return_val = self.return_proj(out[:, 0, :, :])
        direction_val = self.direction_proj(out[:, 1, :, :])

        return return_val, direction_val