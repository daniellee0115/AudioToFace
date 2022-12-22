"""SIMONe-Based Video Generator Implementation"""

from functools import partial
from typing import List

import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.nn import Conv2d
from torch import nn
from torch.nn import TransformerEncoder
from torch.nn import TransformerEncoderLayer
from torch import Tensor
import torch
import numpy as np

from simone.encoder import EncoderConv, EncoderTransformer
from simone.mlp import FeatureMLP
from simone.decoder import Decoder


class SiMONE(nn.Module):
    def __init__(self, xy_resolution=125, in_channels=3, conv_channels=128, conv_stride=2, transformer_channels=2, transformer_layers=1, latent_channels=32):
        """
        Args:
            - xy_resolution: pixel resolution in x and y directions
            - in_channels: number of input channels for each pixel (3 for RGB)
            - conv_channels: number of channels for convolutional layers
            - conv_stride: stride for convolutional layers
            - transformer_channels: number of channels for transformer layer
            - transformer_layers: number of layers for encoder transformer
            - latent_channels: number of channels for featureMLP
        """
        super().__init__()

        self.encoder = EncoderConv(xy_resolution, conv_channels, conv_stride, in_channels)
        conv_resolution = xy_resolution // conv_stride ** 2     # 2 = conv_layers
        transformer_resolution = conv_resolution // 2
        self.transformer_encoder = EncoderTransformer(conv_channels, transformer_channels, transformer_layers, conv_resolution, transformer_resolution)

        # add transformer layer
        decoder_layer = nn.TransformerDecoderLayer(d_model=transformer_channels, nhead=2, dim_feedforward=1024, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)

        # decoder
        K = transformer_resolution ** 2
        self.mlp = FeatureMLP(transformer_channels, latent_channels, transformer_resolution)
        self.decoder = Decoder(xy_resolution, latent_channels, K)


    def forward(self, x, sound_embeddings, maintain_resolution = True):
        T = x.shape[1]

        # encoder
        conv_embeddings = self.encoder(x)
        transformer_embeddings = self.transformer_encoder(conv_embeddings)

        # factor in other embeddings
        sound_embeddings = rearrange(sound_embeddings, "b t (e1 e2) -> b (t e1) e2", e2=2)
        transformer_embeddings = self.transformer_decoder(transformer_embeddings, sound_embeddings)

        # decoder
        spatial_embeddings, temporal_embeddings = self.mlp(transformer_embeddings, T)
        pixels = self.decoder( spatial_embeddings, temporal_embeddings, T, maintain_resolution)

        return pixels