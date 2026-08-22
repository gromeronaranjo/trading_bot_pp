import torch
from models.stockformer import *
from dataclasses import dataclass


@dataclass
class StockFormerConfig:
    n_stocks: int = 23
    n_features: int = 362
    dense_size: int = 128
    n_heads: int = 1
    n_encoder_blocks: int = 2
    dropout: float = 0.2
    time_steps: int = 20
    prediction_steps: int = 2
    kernel_size: int = 2
    learning_rate: float = 0.001
    lr_decay: float = 0.1
    batch_size: int = 16
    epochs: int = 100
    classification_loss_weight: float = 2.0
    save_every_steps: int = 100


def process(stockformer, split, data_x_dir, x_dir, low_y_dir, high_y_dir, batch_size=64):
    data = torch.load(data_x_dir, map_location="cpu")

    num = int(len(data) * split)
    data = data[:num]

    device = next(stockformer.parameters()).device
    stockformer.eval()

    x_values = []
    low_values = []
    high_values = []

    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            batch = data[start:start + batch_size].to(device)

            low, high = stockformer.decoupling(batch)

            low_latent = stockformer.feature_proj(low)
            high_latent = stockformer.feature_proj(high)

            x = batch[..., :2]

            x_values.append(
                x.reshape(-1, 2).cpu()
            )

            low_values.append(
                low_latent.reshape(-1, stockformer.config.dense_size).cpu()
            )

            high_values.append(
                high_latent.reshape(-1, stockformer.config.dense_size).cpu()
            )

    x_tensor = torch.cat(x_values, dim=0)
    low_y_tensor = torch.cat(low_values, dim=0)
    high_y_tensor = torch.cat(high_values, dim=0)

    torch.save(x_tensor, x_dir)
    torch.save(low_y_tensor, low_y_dir)
    torch.save(high_y_tensor, high_y_dir)

    return x_tensor, low_y_tensor, high_y_tensor


config = StockFormerConfig()

stockformer = StockFormer(config)

checkpoint_dir = "/Users/gromeronaranjo/Desktop/personal_project/logs/checkpoints/best.pt"

stockformer.load_state_dict(
    torch.load(checkpoint_dir, map_location="cpu")
)

stockformer.eval()

split = 0.3

data_x_dir = "/Users/gromeronaranjo/Desktop/personal_project/data/X.pt"
x_dir = "/Users/gromeronaranjo/Desktop/personal_project/data/latent_model_x.pt"
low_y_dir = "/Users/gromeronaranjo/Desktop/personal_project/data/latent_model_low_y.pt"
high_y_dir = "/Users/gromeronaranjo/Desktop/personal_project/data/latent_model_high_y.pt"

x_tensor, low_y_tensor, high_y_tensor = process(
    stockformer,
    split,
    data_x_dir,
    x_dir,
    low_y_dir,
    high_y_dir,
    batch_size=64
)

print("x:", x_tensor.shape)
print("low:", low_y_tensor.shape)
print("high:", high_y_tensor.shape)
print("done")