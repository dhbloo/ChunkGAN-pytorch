import torch
from torch import nn
from .common import ResnetBlock
from .encoder import EncoderNetwork
from .decoder import DecoderNetwork
from utils.misc import AttributeDict


class SynthesisNetwork(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 base_channels=16,
                 num_downsample_layers=4,
                 num_intermediate_layers=6,
                 padding_type='reflect',
                 norm_layer=nn.BatchNorm2d,
                 activation=nn.ReLU):
        super(SynthesisNetwork, self).__init__()

        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, base_channels, kernel_size=7),
            norm_layer(base_channels),
            activation()
        ]

        # downsample
        for i in range(num_downsample_layers):
            channels = base_channels * (2**i)
            layers += [
                nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1),
                norm_layer(channels * 2),
                activation()
            ]

        # resnet block
        intermediate_in_channels = base_channels * (2**num_downsample_layers)
        for i in range(num_intermediate_layers):
            layers += [
                ResnetBlock(intermediate_in_channels,
                            padding_type=padding_type,
                            norm_layer=norm_layer,
                            activation=activation)
            ]

        # upsample
        for i in range(num_downsample_layers):
            channels = base_channels * (2**(num_downsample_layers - 1 - i))
            layers += [
                nn.ConvTranspose2d(channels * 2,
                                   channels,
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1),
                norm_layer(channels),
                activation()
            ]

        # output
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(base_channels, out_channels, kernel_size=7),
            nn.Tanh()
        ]

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        return x


class Generator(nn.Module):
    def __init__(self,
                 image_size,
                 chunk_size,
                 out_channels,
                 feature_channels=16,
                 bg_latent_dim=512,
                 chunk_latent_dim=512,
                 compose_func='sum',
                 G_bg_args={},
                 G_chunk_args={},
                 synthesis_args={}):
        super(Generator, self).__init__()

        self.image_size = image_size
        self.chunk_size = chunk_size
        self.bg_latent_dim = bg_latent_dim
        self.chunk_latent_dim = chunk_latent_dim
        self.feature_channels = feature_channels
        self.G_bg = DecoderNetwork(image_size, feature_channels, bg_latent_dim, **{
            "num_layers": 4,
            "duplicate_layer_set": [3],
            **G_bg_args
        })
        self.G_chunk = DecoderNetwork(chunk_size, feature_channels, chunk_latent_dim,
                                      **G_chunk_args)
        self.synthesis = SynthesisNetwork(feature_channels, out_channels, **synthesis_args)
        if compose_func == 'replace':
            self.compose_func = 'replace'
        elif compose_func == 'max':
            self.compose_func = torch.maximum
        elif compose_func == 'sum':
            self.compose_func = torch.add
        else:
            raise NotImplementedError(f'compose function [{compose_func}] not implemented')

    def forward(self, bg_latent, chunk_latents, chunk_trans, no_background=False):
        assert len(chunk_latents) == len(chunk_trans)
        bg_feature = self.G_bg(bg_latent)
        if no_background:
            bg_feature = bg_feature.mul(0)

        combined_feature = []
        for batch_idx in range(len(chunk_latents)):
            batch_chunk_latents = chunk_latents[batch_idx]
            batch_chunk_trans = chunk_trans[batch_idx]

            batch_combined_feature = bg_feature[batch_idx:batch_idx + 1, :, :, :]
            if len(batch_chunk_trans) > 0:
                assert len(batch_chunk_latents) == len(batch_chunk_trans)
                batch_chunk_feature = self.G_chunk(batch_chunk_latents)
                for i in range(len(batch_chunk_trans)):
                    x, y, w, h = batch_chunk_trans[i]
                    chunk_feature = batch_chunk_feature[i:i + 1, :, :, :]
                    scaled_chunk_feature = nn.functional.interpolate(chunk_feature, (h, w),
                                                                     mode='bilinear',
                                                                     align_corners=False)

                    if self.compose_func == 'replace':
                        batch_combined_feature = batch_combined_feature.clone()
                        batch_combined_feature[:, :, y:y + h, x:x + w] = scaled_chunk_feature
                    else:
                        translated_feature = torch.zeros_like(bg_feature[:1, :, :, :])
                        translated_feature[:, :, y:y + h, x:x + w] = scaled_chunk_feature
                        batch_combined_feature = self.compose_func(batch_combined_feature,
                                                                   translated_feature)
            combined_feature += [batch_combined_feature]

        combined_feature = torch.cat(combined_feature, dim=0)
        image = self.synthesis(combined_feature)
        return image
