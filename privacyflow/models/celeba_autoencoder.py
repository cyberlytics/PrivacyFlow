import torch
import torch.nn as nn


class CelebADenoisingAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = self._get_encoder()
        self.decoder = self._get_decoder()

    def _get_encoder(self):
        layers = []
        layers.append(nn.Conv2d(3,32, kernel_size=3,padding=1))
        layers.append(nn.ReLU(True))
        layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
        layers.append(nn.Conv2d(32, 64, kernel_size=3, padding=1))
        layers.append(nn.ReLU(True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(nn.Conv2d(64, 128, kernel_size=3, padding=1))
        layers.append(nn.ReLU(True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(nn.Conv2d(128, 256, kernel_size=3, padding=1))
        layers.append(nn.ReLU(True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)
    def _get_decoder(self):
        layers = []
        layers.append(nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2))
        layers.append(nn.ReLU(True))
        layers.append(nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2))
        layers.append(nn.ReLU(True))
        layers.append(nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2))
        layers.append(nn.ReLU(True))
        layers.append(nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2))
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
