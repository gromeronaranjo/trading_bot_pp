import os
import torch
import torch.nn.functional as F
from dataclasses import dataclass
import torch.optim as optim
from models.latentdynamics import *


@dataclass
class LatentDynamicsConfig:
    dense_size: int = 128
    batch_size: int = 256
    epochs: int = 150
    learning_rate: float = 3e-4
    test_split: float = 0.2
    evaluation_every_steps: int = 100


def evaluate(model, x, y, config, device):
    model.eval()
    total_loss = 0.0
    batches = 0

    with torch.no_grad():
        for start in range(0, len(x), config.batch_size):
            batch_x = x[start:start + config.batch_size].to(device)
            batch_y = y[start:start + config.batch_size].to(device)
            loss = F.mse_loss(model(batch_x[:, 0:1], batch_x[:, 1:2]), batch_y)
            total_loss += loss.item()
            batches += 1

    return total_loss / batches


config = LatentDynamicsConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

latent_model_x = torch.load("/Users/gromeronaranjo/Desktop/personal_project/data/latent_model_x.pt", map_location="cpu")
latent_model_y = torch.load("/Users/gromeronaranjo/Desktop/personal_project/data/latent_model_y.pt", map_location="cpu")

split = int(len(latent_model_x) * (1 - config.test_split))

train_x, test_x = latent_model_x[:split], latent_model_x[split:]
train_y, test_y = latent_model_y[:split], latent_model_y[split:]

model = LatentDynamicsModel(config).to(device)
optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)

logs_directory = "/Users/gromeronaranjo/Desktop/personal_project/logs"
checkpoint_directory = os.path.join(logs_directory, "checkpoints")
os.makedirs(checkpoint_directory, exist_ok=True)

steps = []
train_losses = []
test_losses = []

best_test_loss = float("inf")
global_step = 0
train_total = 0.0
train_batches = 0

print("train samples:", len(train_x))
print("test samples:", len(test_x))
print("device:", device)

for epoch in range(config.epochs):
    model.train()
    permutation = torch.randperm(len(train_x))

    for start in range(0, len(train_x), config.batch_size):
        indices = permutation[start:start + config.batch_size]
        x = train_x[indices].to(device)
        y = train_y[indices].to(device)

        optimizer.zero_grad()
        loss = F.mse_loss(model(x[:, 0:1], x[:, 1:2]), y)
        loss.backward()
        optimizer.step()

        global_step += 1
        train_total += loss.item()
        train_batches += 1

        print(f"epoch {epoch + 1}/{config.epochs} | step {global_step} | train loss {loss.item():.8f}")

        if global_step % config.evaluation_every_steps == 0:
            train_loss = train_total / train_batches
            test_loss = evaluate(model, test_x, test_y, config, device)

            steps.append(global_step)
            train_losses.append(train_loss)
            test_losses.append(test_loss)

            torch.save({"steps": steps, "train_loss": train_losses, "test_loss": test_losses}, os.path.join(logs_directory, "latent_dynamics_losses.pt"))

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save(model.state_dict(), os.path.join(checkpoint_directory, "latent_dynamics_best.pt"))
                print("saved new best model")

            print(f"step {global_step} evaluation | train loss {train_loss:.8f} | test loss {test_loss:.8f} | best test loss {best_test_loss:.8f}")

            train_total = 0.0
            train_batches = 0
            model.train()

print("done")