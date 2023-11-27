import torch
import torch.nn as nn


class SAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 800),
            torch.nn.ReLU(),
            torch.nn.Linear(800, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 50)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(50, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 800),
            torch.nn.ReLU(),
            torch.nn.Linear(800, 28*28),
            torch.nn.Sigmoid()
        )