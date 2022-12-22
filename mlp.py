"""MLP Implementation for SIMONe-Based Video Generator"""

from typing import List
from einops import rearrange
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn
from torch.nn import Conv2d


class MLP(nn.Module):
    """Create a MLP with `len(hidden_features)` hidden layers, each with `hidden_features[i]` features."""

    def __init__(
        self, in_features: int, out_features: int, hidden_features: List[int]
    ):
        super().__init__()

        layers = []
        last_size = in_features

        for size in hidden_features:
            layers.append(nn.Linear(last_size, size))
            last_size = size
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(last_size, out_features))

        self.sequential = torch.nn.Sequential(*layers)

    def forward(self, x: Tensor):
        return self.sequential(x)


class FeatureMLP(nn.Module):
    def __init__(self, transformer_channels, latent_channels, resolution):
        super().__init__()
        self.transformer_channels = transformer_channels
        self.latent_channels = latent_channels
        self.resolution = resolution
        self.K = resolution ** 2

        self.spatial_mlp = MLP(
            in_features=transformer_channels, out_features=latent_channels * 2, hidden_features=[1024]
        )
        self.temporal_mlp = MLP(
            in_features=transformer_channels, out_features=latent_channels * 2, hidden_features=[1024]
        )

    def forward(self, x: Tensor, T):
        x = rearrange(x, "b (t h w) c -> b t h w c", t=T, h=self.resolution, w=self.resolution, c=self.transformer_channels)

        spatial_features = torch.mean(x, dim=1)
        temporal_features = torch.mean(x, dim=(2, 3))

        spatial_features = rearrange(spatial_features, "b h w c -> (b h w) c")
        temporal_features = rearrange(temporal_features, "b t c -> (b t) c", t=T, c=self.transformer_channels)

        # apply MLPs
        spatial_features = self.spatial_mlp(spatial_features)
        temporal_features = self.temporal_mlp(temporal_features)

        # reshape on a per-feature basis
        spatial_features = rearrange(spatial_features, "(b k) (c c2) -> b k c c2", k=self.K, c=self.latent_channels, c2=2)
        temporal_features = rearrange(temporal_features, "(b t) (c c2) -> b t c c2", t=T, c=self.latent_channels, c2=2)

        return spatial_features, temporal_features