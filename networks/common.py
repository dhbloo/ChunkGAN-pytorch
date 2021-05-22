import torch
from torch import nn


class ResnetBlock(nn.Module):
    def __init__(self,
                 channels,
                 dropout_p=0,
                 norm_layer=nn.BatchNorm2d,
                 activation=nn.ReLU,
                 padding_type='reflect',
                 kernel_size=3):
        super(ResnetBlock, self).__init__()

        layers = []
        padding_size = (kernel_size - 1) // 2

        conv_padding_size = 0
        if padding_type == 'reflect':
            layers += [nn.ReflectionPad2d(padding_size)]
        elif padding_type == 'replicate':
            layers += [nn.ReplicationPad2d(padding_size)]
        elif padding_type == 'zero':
            conv_padding_size = padding_size
        else:
            raise NotImplementedError(f'padding [{padding_type}] not implemented')

        layers += [
            nn.Conv2d(channels, channels, kernel_size, 1, conv_padding_size),
            norm_layer(channels),
            activation()
        ]

        if dropout_p > 0:
            layers += [nn.Dropout2d(dropout_p)]

        conv_padding_size = 0
        if padding_type == 'reflect':
            layers += [nn.ReflectionPad2d(padding_size)]
        elif padding_type == 'replicate':
            layers += [nn.ReplicationPad2d(padding_size)]
        elif padding_type == 'zero':
            conv_padding_size = padding_size
        else:
            raise NotImplementedError(f'padding [{padding_type}] not implemented')

        layers += [
            nn.Conv2d(channels, channels, kernel_size, 1, conv_padding_size),
            norm_layer(channels)
        ]

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        x = x + self.conv_block(x)
        return x
