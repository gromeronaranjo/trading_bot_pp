import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as functional

class LatentDynamicsModel(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, x, y):
        pass