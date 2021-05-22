import torch
from torch import nn
from .common import ResnetBlock


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activition = nn.LeakyReLU(1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activition(x)
        return x


class EncoderNetwork(nn.Module):
    def __init__(self,
                 image_size,
                 in_channels,
                 use_labelmap=False,
                 latent_dim=512,
                 base_channels=16,
                 num_layers=3):
        super(EncoderNetwork, self).__init__()

        total_layers = num_layers + 1
        assert (image_size % (2**total_layers) == 0) and (image_size > 0)

        self.image_size = image_size
        self.use_labelmap = use_labelmap
        self.latent_dim = latent_dim
        input_channels = in_channels + self.use_labelmap
        conv_layers = [EncoderBlock(input_channels, base_channels, 4, 2, 1)]  # first encoder layer
        for i in range(num_layers):
            channels = base_channels * (2**i)
            conv_layers += [
                ResnetBlock(channels),
                EncoderBlock(channels, channels * 2, kernel_size=4, stride=2, padding=1)
            ]

        conv_out_channels = base_channels * (2**num_layers)
        conv_layers += [ResnetBlock(conv_out_channels)]
        self.conv = nn.Sequential(*conv_layers)

        conv_out_features = (image_size // (2**total_layers))**2
        self.fc = nn.Linear(conv_out_features * conv_out_channels, self.latent_dim)

    def forward(self, x, labelmap=None):
        if self.use_labelmap:
            assert labelmap is not None
            x = torch.cat((x, labelmap), 1)
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
