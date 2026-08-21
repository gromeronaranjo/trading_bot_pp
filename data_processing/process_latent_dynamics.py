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


def process(stockformer, split, data_x_dir, x_dir, y_dir, batch_size=4096):
    data = torch.load(data_x_dir, map_location="cpu")

    n_sequences, sequence_length, n_stocks, features = data.shape

    data = data.reshape(
        n_sequences * sequence_length * n_stocks,
        features
    )

    num = int(data.size(0) * split)
    data = data[:num]

    x_tensor = data[:, :2].float()

    device = next(stockformer.parameters()).device
    stockformer.eval()

    y = []

    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            batch = data[start:start + batch_size].to(device)

            latent = stockformer.feature_proj(batch)

            y.append(latent.cpu())

    y_tensor = torch.cat(y, dim=0)

    torch.save(x_tensor, x_dir)
    torch.save(y_tensor, y_dir)

    return x_tensor, y_tensor

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
y_dir = "/Users/gromeronaranjo/Desktop/personal_project/data/latent_model_y.pt"

x_tensor, y_tensor = process(stockformer, split, data_x_dir, x_dir, y_dir, batch_size=4096)
print("done")