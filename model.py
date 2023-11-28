import torch
import torch.nn as nn


class StackedAutoencoder(nn.Module):
    def __init__(self, bottleneck_size):
        super(StackedAutoencoder, self).__init__()
        self.encoded_feature_size = bottleneck_size
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 800),
            nn.ReLU(),
            nn.Linear(800, 200),
            nn.ReLU(),
            nn.Linear(200, self.encoded_feature_size),  # Bottleneck layer
            nn.ReLU()
        )
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(self.encoded_feature_size, 200),
            nn.ReLU(),
            nn.Linear(200, 800),
            nn.ReLU(),
            nn.Linear(800, 28*28),
            nn.Sigmoid(),
            nn.Unflatten(1, (28, 28))
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class Classifier(nn.Module):
    def __init__(self, encoded_feature_size, num_classes):
        super(Classifier, self).__init__()
        self.encoder = StackedAutoencoder(encoded_feature_size).encoder  # Reuse the encoder
        self.classifier = nn.Sequential(
            nn.Linear(encoded_feature_size, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        output = self.classifier(encoded)
        return output



if __name__=="__main__":
    B, C, H, W = 1, 1, 28, 28
    model = SAE()

    image = torch.randn(B, C, H, W)
    image = image.reshape(B, C, -1)
    
    out = model(image)
    out = out.reshape(B, C, H, W)
    print(out.shape)