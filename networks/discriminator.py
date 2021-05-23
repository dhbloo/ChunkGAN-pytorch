import torch
from torch import nn
from utils.misc import suppress_tracer_warnings


class MinibatchStdLayer(nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        with suppress_tracer_warnings():  # as_tensor results are registered as constants
            G = torch.min(torch.as_tensor(self.group_size),
                          torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F

        y = x.reshape(
            G, -1, F, c, H, W
        )  # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)  # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)  # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()  # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2, 3, 4])  # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)  # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)  # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)  # [NCHW]   Append to input as new channels.
        return x


class Discriminator(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            base_channels=128,
            num_layers=2,
            duplicate_layer_set=[1],
            norm_layer=nn.InstanceNorm2d,
            activation=nn.LeakyReLU,
            mbstd_group_size=32,  # Group size for the minibatch standard deviation layer, None = entire minibatch.
            mbstd_num_channels=1,  # Number of features for the minibatch standard deviation layer, 0 = disable.
    ):
        super(Discriminator, self).__init__()

        total_layers = num_layers + 1 + len(
            [l for l in range(num_layers) if l in duplicate_layer_set])
        assert (image_size % (2**total_layers) == 0) and (image_size > 0)

        self.image_size = image_size

        self.mbstd = MinibatchStdLayer(
            group_size=mbstd_group_size,
            num_channels=mbstd_num_channels) if mbstd_num_channels > 0 else None

        conv_layers = [
            nn.Conv2d(in_channels + mbstd_num_channels,
                      base_channels,
                      kernel_size=4,
                      stride=2,
                      padding=1),
            activation()
        ]
        for i in range(num_layers):
            channels = base_channels * (2**i)
            conv_layers += [
                nn.Conv2d(channels, channels * 2, kernel_size=4, stride=2, padding=1),
                norm_layer(channels * 2),
                activation()
            ]

            if i in duplicate_layer_set:
                conv_layers += [
                    nn.Conv2d(channels * 2, channels * 2, kernel_size=4, stride=2, padding=1),
                    norm_layer(channels * 2),
                    activation()
                ]

        conv_out_channels = base_channels * (2**num_layers)
        conv_layers += [
            nn.Conv2d(conv_out_channels, 1, kernel_size=3, stride=1, padding=1),
        ]

        self.conv = nn.Sequential(*conv_layers)

        # conv_out_features = (image_size // (2**total_layers))**2
        # fc_layers = [
        #     nn.Linear(conv_out_features * conv_out_channels, conv_out_channels),
        #     activation(),
        #     nn.Linear(conv_out_channels, 1)
        # ]
        # self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        if self.mbstd is not None:
            x = self.mbstd(x)
        # For PatchGAN, outputs a matrix of logits
        x = self.conv(x)
        # x = x.view(x.shape[0], -1)
        # x = self.fc(x)
        return x
