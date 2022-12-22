"""MLP-Based Decoder for SIMONe-inspired Video Generator"""

from typing import List
import random

from einops import rearrange, repeat
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn
from torch.nn import Conv2d

from simone.mlp import MLP


def get_position_encoding(range: Tensor, target_shape: List[int], dim: int):
    """Create a tensor of shape `target_shape` that is filled with values from `range` along `dim`."""
    assert len(range.shape) == 1
    assert len(range) == target_shape[dim]

    view_shape = [1 for _ in target_shape]
    view_shape[dim] = target_shape[dim]
    range = range.view(view_shape)
    encoding = range.expand(target_shape)
    assert encoding.shape == tuple(target_shape)
    return encoding


def build_t_encoding(T, desired_shape: List[int], time_indexes: Tensor, device, dtype):
    # Form the T indicator
    t_linspace = torch.linspace(0, 1, T, device=device, dtype=dtype)
    t_linspace = t_linspace.index_select(dim=0, index=time_indexes)
    t_encoding = get_position_encoding(t_linspace, desired_shape, dim=1)

    return t_encoding


class Decoder(nn.Module):
    def __init__(self, xy_resolution, latent_channels, K):
        super().__init__()
        self.resolution = xy_resolution
        self.latent_channels = latent_channels
        self.K = K

        self.mlp = MLP(in_features=K*(latent_channels*2+1), out_features=xy_resolution*xy_resolution*3, hidden_features=[512 for _ in range(5)])


    def forward(self, object_latents: Tensor, temporal_latents: Tensor, T):
        batch_size = object_latents.shape[0]
        K = self.K
        
        assert object_latents.shape == (batch_size, K, self.latent_channels, 2)
        assert temporal_latents.shape == (batch_size, T, self.latent_channels, 2)

        time_indexes = torch.tensor(sorted(random.sample(range(T), T)), device=object_latents.device)
        
        object_latents = object_latents.mean(-1, False)
        temporal_latents = temporal_latents.mean(-1, False)

        # Expand the latents to the full prediction size, so that we get a unique representation for each pixel
        object_latents = repeat(object_latents, "b k c -> b t k c", t=T)
        temporal_latents = repeat(temporal_latents, "b td c -> b td k c", k=K)
       
        # Build x y t indicator features
        desired_shape = [batch_size, T, K, 1]
        t_encoding = build_t_encoding(T, desired_shape, time_indexes, object_latents.device, object_latents.dtype)

        # Concatenate features together
        x = torch.cat([object_latents, temporal_latents, t_encoding], dim=-1)

        # MLP feature decoder
        x = rearrange(x, "b t k c -> (b t) (k c)", c=2 * self.latent_channels+1)
        x = self.mlp(x)
        x = rearrange(x, "(b t) (h w c) -> b t c h w", t=T, c=3, h=self.resolution, w=self.resolution)

        return x 