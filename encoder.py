"""Convolutional Encoder and Transformer for SIMONe-Based Video Generator"""

from einops import rearrange
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn
from torch.nn import Conv2d


class EncoderConv(nn.Module):
    """2-Layered Convolutional Encoder"""
    def __init__(self, xy_resolution, conv_channels=128, conv_stride=2, in_channels=3):
        super().__init__()
        torch.set_flush_denormal(True)
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.stride = 2
        self.layers = 2
        self.resolution = xy_resolution
        self.new_resolution = xy_resolution // conv_stride ** self.layers

        self.conv_1 = Conv2d(in_channels=3, out_channels=conv_channels, kernel_size=4, stride=conv_stride, padding=(1, 1))
        self.conv_2 = Conv2d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=4, stride=conv_stride, padding=(1, 1))

    def forward(self, x: Tensor):
        T = x.shape[1]
        x = rearrange(x, "b t c h w -> (b t) c h w", t=T, c=self.in_channels, h=self.resolution, w=self.resolution)

        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        
        return rearrange(x, "(b t) c h w -> b t c h w", t=T, c=self.conv_channels, w=self.new_resolution, h=self.new_resolution)


class EncoderTransformer(nn.Module):
    def __init__(self, conv_channels, transformer_channels, transformer_layers: int, resolution, new_resolution):
        super().__init__()
        self.resolution = resolution
        self.new_resolution = new_resolution
        self.transformer_channels = transformer_channels
        self.conv_channels = conv_channels

        # this template layer will get cloned inside the TransformerEncoder modules below.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_channels,
            nhead=2,
            dim_feedforward=1024,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.linear_layer = torch.nn.Linear(
            in_features=conv_channels, out_features=transformer_channels, bias=False
        )

        self.transformer_1 = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=transformer_layers)
        

    def forward(self, x: Tensor):
        batch_size = x.shape[0]
        T = x.shape[1]
        x = rearrange(x, "b t c h w -> b t h w c", b=batch_size, t=T, h=self.resolution, w=self.resolution, c=self.conv_channels) 

        # apply linear transformation to project ENCODER_CONV_CHANNELS to TRANSFORMER_CHANNELS
        x = self.linear_layer(x)

        # apply transformer
        x = rearrange(x, "b t h w c -> b (t h w) c", b=batch_size, t=T, h=self.resolution, w=self.resolution, c=self.transformer_channels) 
        x = self.transformer_1(x)
        x = rearrange(x, "b (t h w) c -> (b t) c h w", b=batch_size, t=T, h=self.resolution, w=self.resolution, c=self.transformer_channels) 

        x = F.avg_pool2d(x, kernel_size=2) * 2
        x = rearrange(x, "(b t) c h w -> b (t h w) c", b=batch_size, t=T, h=self.new_resolution, w=self.new_resolution, c=self.transformer_channels) 

        return x