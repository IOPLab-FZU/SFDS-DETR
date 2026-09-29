"""Frequency-Aware Feature Extraction Backbone (FAFEB) building blocks.

The names in this module follow the SFDS-DETR paper. The operations are a
hierarchical regrouping of the original implementation and preserve its
forward computation.
"""

import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward

from ultralytics.nn.modules.block import C2f
from ultralytics.nn.modules.conv import Conv


class WaveletTransform(nn.Module):
    """First-level Haar transform producing LL, HL, LH and HH sub-bands."""

    def __init__(self, in_ch=None, out_ch=None):
        super().__init__()
        self.dwt = DWTForward(J=1, mode="zero", wave="haar")

    def forward(self, x):
        transform = getattr(self, "dwt", None) or self.wt  # ``wt`` supports legacy pickled models.
        ll, high = transform(x)
        return torch.cat((ll, high[0][:, :, 0], high[0][:, :, 1], high[0][:, :, 2]), dim=1)


class CascadedWaveletTransform(nn.Module):
    """Apply the next Haar transform to the preceding three-channel LL band."""

    def __init__(self, in_ch=None, out_ch=None):
        super().__init__()
        self.dwt = DWTForward(J=1, mode="zero", wave="haar")

    def forward(self, x):
        transform = getattr(self, "dwt", None) or self.wt  # ``wt`` supports legacy pickled models.
        ll, high = transform(x[:, :3])
        return torch.cat((ll, high[0][:, :, 0], high[0][:, :, 1], high[0][:, :, 2]), dim=1)


class SE(nn.Module):
    """Squeeze-and-excitation used in the CAAE frequency-alignment path."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch, channels, _, _ = x.shape
        weights = self.fc(self.avg_pool(x).view(batch, channels)).view(batch, channels, 1, 1)
        return x * weights.expand_as(x)


class CEM(nn.Module):
    """Channel Exchange Modulation for adaptive spatial-frequency fusion."""

    def __init__(self, in_channels, channels, reduction=8):
        super().__init__()
        self.branches = len(in_channels)
        hidden = max(int(channels / reduction), 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden, channels * self.branches, 1, bias=False),
        )
        self.softmax = nn.Softmax(dim=1)
        self.projections = nn.ModuleList(
            Conv(c, channels, 1) if c != channels else nn.Identity() for c in in_channels
        )

    def forward(self, features):
        projections = getattr(self, "projections", None) or self.conv1x1
        branches = getattr(self, "branches", None) or self.height
        features = [projection(feature) for projection, feature in zip(projections, features)]
        batch, channels, height, width = features[0].shape
        features = torch.cat(features, dim=1).view(batch, branches, channels, height, width)
        attention = self.mlp(self.avg_pool(features.sum(dim=1)))
        attention = self.softmax(attention.view(batch, branches, channels, 1, 1))
        return (features * attention).sum(dim=1)


class CAAE(nn.Module):
    """Channel Attention Alignment Enhancement module from FAFEB.

    The frequency feature is aligned by a 3x3 local convolution, a 1x1
    point-wise projection and SE, then injected into the spatial feature by
    two consecutive CEM modules. This exactly groups the five operations used
    by the original YAML without changing their order or tensor shapes.
    """

    def __init__(self, in_channels, out_channels, alignment_channels, reduction=8):
        super().__init__()
        frequency_channels, spatial_channels = in_channels
        self.frequency_conv = Conv(frequency_channels, alignment_channels, 3, 1)
        self.pwconv = Conv(alignment_channels, out_channels, 1, 1)
        self.se = SE(out_channels)
        self.cem1 = CEM((out_channels, spatial_channels), out_channels, reduction)
        self.cem2 = CEM((out_channels, spatial_channels), out_channels, reduction)

    def forward(self, features):
        frequency, spatial = features
        frequency = self.se(self.pwconv(self.frequency_conv(frequency)))
        enhanced = self.cem1((frequency, spatial))
        return self.cem2((enhanced, spatial))


class ConvBlock(C2f):
    """Spatial-path ConvBlock shown in the FAFEB diagram (C2f implementation)."""

