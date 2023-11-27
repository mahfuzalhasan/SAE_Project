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
    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return dec

if __name__=="__main__":
    B, C, H, W = 1, 1, 28, 28
    model = SAE()

    image = torch.randn(B, C, H, W)
    image = image.reshape(B, C, -1)
    
    out = model(image)
    out = out.reshape(B, C, H, W)
    print(out.shape)