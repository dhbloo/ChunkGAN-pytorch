import torch
from torch import nn
from .common import ResnetBlock


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding=0):
        super(DecoderBlock, self).__init__()

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,
                                       output_padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activition = nn.LeakyReLU(1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activition(x)
        return x


class DecoderNetwork(nn.Module):
    def __init__(self, image_size, out_channels, latent_dim=512, base_channels=16, num_layers=3, duplicate_layer_set=[2]):
        super(DecoderNetwork, self).__init__()

        total_layers = num_layers + len([l for l in range(num_layers) if l in duplicate_layer_set])
        assert (image_size % (2**total_layers) == 0) and (image_size > 0)

        self.image_size = image_size
        self.latent_dim = latent_dim
        self.latent_size = image_size // (2**total_layers)
        self.conv_in_channels = base_channels * (2**num_layers)
        conv_in_features = self.latent_size**2
        self.fc = nn.Linear(self.latent_dim, self.conv_in_channels * conv_in_features)

        conv_layers = [ResnetBlock(self.conv_in_channels)]
        for i in range(num_layers):
            channels = base_channels * (2**(num_layers - 1 - i))
            conv_layers += [
                DecoderBlock(channels * 2, channels, kernel_size=4, stride=2, padding=1),
                ResnetBlock(channels)
            ]

            if i in duplicate_layer_set:
                conv_layers += [
                DecoderBlock(channels, channels, kernel_size=4, stride=2, padding=1),
                ResnetBlock(channels)
            ]

        conv_layers += [
            nn.ReflectionPad2d(2),
            nn.Conv2d(base_channels, out_channels, kernel_size=5)
        ]

        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], self.conv_in_channels, self.latent_size, self.latent_size)
        x = self.conv(x)
        return x

