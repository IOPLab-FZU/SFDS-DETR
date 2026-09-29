"""Orientation-Sensitive Pyramid Fusion Encoder (OSPFE) building blocks."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.block import C2f
from ultralytics.nn.modules.conv import Conv


class OSRConv(nn.Module):
    """Orientation-Sensitive Re-parameterized Convolution (OSR-Conv).

    Training uses parallel 3x3, 1x1, 1x3 and 3x1 branches. Inference can fuse
    them into one equivalent 3x3 convolution without additional cost.
    """

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True, bn=False, deploy=False):
        super().__init__()
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.bn = nn.BatchNorm2d(c1) if bn and c2 == c1 and s == 1 else None
        self.conv3x3 = Conv(c1, c2, k=3, s=s, p=1, g=g, act=False)
        self.conv1x1 = Conv(c1, c2, k=1, s=s, p=0, g=g, act=False)
        self.conv1x3 = Conv(c1, c2, k=(1, 3), s=s, p=(0, 1), g=g, act=False)
        self.conv3x1 = Conv(c1, c2, k=(3, 1), s=s, p=(1, 0), g=g, act=False)

    def forward(self, x):
        identity = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv3x3(x) + self.conv1x1(x) + self.conv1x3(x) + self.conv3x1(x) + identity)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv3x3)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv1x1)
        kernel1x3, bias1x3 = self._fuse_bn_tensor(self.conv1x3)
        kernel3x1, bias3x1 = self._fuse_bn_tensor(self.conv3x1)
        kernel_id, bias_id = self._fuse_bn_tensor(self.bn)
        kernel = (
            kernel3x3
            + F.pad(kernel1x1, [1, 1, 1, 1])
            + F.pad(kernel1x3, [0, 0, 1, 1])
            + F.pad(kernel3x1, [1, 1, 0, 0])
            + kernel_id
        )
        return kernel, bias3x3 + bias1x1 + bias1x3 + bias3x1 + bias_id

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel = torch.zeros((self.c1, input_dim, 3, 3), dtype=branch.weight.dtype, device=branch.weight.device)
                for i in range(self.c1):
                    kernel[i, i % input_dim, 1, 1] = 1
                self.id_tensor = kernel
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        else:
            return 0, 0
        std = (running_var + eps).sqrt()
        scale = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * scale, beta - running_mean * gamma / std

    def fuse_convs(self):
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        source = self.conv3x3.conv
        self.conv = nn.Conv2d(
            source.in_channels,
            source.out_channels,
            kernel_size=3,
            stride=source.stride,
            padding=source.padding,
            dilation=source.dilation,
            groups=source.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for attr in ("conv3x3", "conv1x1", "conv1x3", "conv3x1", "bn"):
            if hasattr(self, attr):
                delattr(self, attr)
        self.forward = self.forward_fuse

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class OSRF(C2f):
    """Orientation-Sensitive Re-parameterized Fusion block."""

    def __init__(self, c1, c2, n=3, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut=shortcut, g=g, e=e)
        self.m = nn.ModuleList(OSRConv(self.c, self.c) for _ in range(n))
