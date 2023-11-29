import torch
import torch.nn as nn

from discriminative import Discriminative


class StackedAutoencoder(nn.Module):
    def __init__(self, bottleneck_size, batch_size, alpha=0.5, k = 11):
        super(StackedAutoencoder, self).__init__()
        self.encoded_feature_size = bottleneck_size
        self.batch_size = batch_size
        self.discriminative = Discriminative(alpha, batch_size, k)
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
        latent = self.encoder(x)
        batch = x.size(0)
        flatten_input = x.reshape(batch, -1)
        non_anchor, anchor = self.discriminative(flatten_input, latent)
        decoded = self.decoder(latent)
        return decoded, non_anchor, anchor, latent

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