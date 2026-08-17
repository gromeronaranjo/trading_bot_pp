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

def pearson_stock_graph(x):
    B, T, N, F = x.shape
    x = x[:, :, :, 0] #(B, T, N)

    x = x.reshape(B * T, N)

    coefs = torch.corrcoef(x.T)

    return coefs


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
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, high):
        B, T, N, D = high.shape
        high = high.permute(0, 2, 3, 1).reshape(B * N, D, T)

        high = functional.pad(high, (1, 0))
        x = functional.relu(self.conv1(high))
        x = self.dropout(x)
        x = functional.pad(x, (2, 0))
        x = functional.relu(self.conv2(x))
        x = self.dropout(x)

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

class TemporalEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj1 = nn.Linear(55, config.dense_size)
        self.proj2 = nn.Linear(config.dense_size, config.dense_size)

    def forward(self, te):
        day_of_week = functional.one_hot(
            te[..., 0].long() % 5,
            num_classes=5
        ).float() #(B, T, 5)

        time_of_day = functional.one_hot(
            te[..., 1].long() % 50,
            num_classes=50
        ).float() #(B, T, 50)

        te = torch.cat([day_of_week, time_of_day], dim=-1)
        te = functional.relu(self.proj1(te))
        te = self.proj2(te)

        return te.unsqueeze(2)
    
class DualFrequencySpatiotemporalEncoder(nn.Module):
    def __init__(self, config, stock_embedding):
        super().__init__()
        self.feature_proj = nn.Linear(config.n_features, config.dense_size)
        self.temporal_attention = MultiHeadAttention(config, Head)
        self.dialated_cnn = DialatedCNN(config)

        self.cross_stock_low = CrossStockAttention(config)
        self.cross_stock_high = CrossStockAttention(config)

        self.register_buffer(
            "stock_embedding",
            stock_embedding.unsqueeze(0).unsqueeze(0) #(1, 1, N, N)
        )

        self.stock_embedding_processing = nn.Linear(config.n_stocks, config.dense_size)
        self.temporal_embedding = TemporalEmbedding(config)

    def forward(self, low, high, te):
        low = self.feature_proj(low)
        high = self.feature_proj(high)

        stock_embedding = self.stock_embedding_processing(self.stock_embedding)
        time_embedding = self.temporal_embedding(te)

        low = low + time_embedding
        high = high + time_embedding
        
        low_out = self.temporal_attention(low)
        high_out = self.dialated_cnn(high)

        low_out = low_out + stock_embedding
        high_out = high_out + stock_embedding

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
    def __init__(self, config, stock_embedding):
        super().__init__()
        self.config = config
        self.decoupling = Decoupling()

        self.duel_freq_spatio_temporal_encoder = DualFrequencySpatiotemporalEncoder(
            config,
            stock_embedding
        )

        self.future_pred_low = FuturePredictor(config)
        self.future_pred_high = FuturePredictor(config)
        self.dual_frequency_fusion = DualFrequencyFusionModule(config)

        self.return_proj = nn.Linear(config.dense_size, 1)
        self.direction_proj = nn.Linear(config.dense_size, 1)

    def forward(self, x, te):
        low, high = self.decoupling(x)
        low, high = self.duel_freq_spatio_temporal_encoder(low, high, te)

        low = self.future_pred_low(low)
        high = self.future_pred_high(high)

        out = self.dual_frequency_fusion(low, high)

        return_val = self.return_proj(out)
        direction_val = self.direction_proj(out)

        return return_val, direction_val