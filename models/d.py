### Code from: https://raw.githubusercontent.com/NVIDIA/vid2vid/master/models/networks.py

### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import spectral_norm
from models.utils import *


# Defines the PatchGAN discriminator with the specified arguments.

def define_D2d(input_nc, ndf, init_gain=0.02, gpu_ids=[device], mode=2):
    """Create a 'PatchGAN' discriminator

    Parameters:
        mode (str)         -- BatchNorm/conv dimension
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator
    """
    net = NLayerDiscriminator(input_nc, ndf)
    return init_net(net, mode, init_gain, gpu_ids)


def define_D3d(input_nc, ndf, init_gain=0.02, gpu_ids=[device], mode=3):
    """Create a 'PatchGAN' discriminator

    Parameters:
        mode (str)         -- BatchNorm/conv dimension
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator
    """
    net = NLayerDiscriminator3d(input_nc, ndf)
    return init_net(net, mode, init_gain, gpu_ids)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__()
        use_bias = True

        kw = 3
        padw = 1
        sequence = [spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias)),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [spectral_norm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input, mask=None):
        """Standard forward."""
        if input.dim() == 5:
            batch_size, _, _, height, width = input.shape
            input = input.view(batch_size, -1, height, width)
        return self.model(input)

class NLayerDiscriminator3d(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d, pooling=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__()
        use_bias = True
        self.pooling = pooling

        kw = (3,3,3)
        padw = (1,1,1)
        sequence = [spectral_norm(nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=(1,2,2), padding=padw)), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                spectral_norm(nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=(1,2,2), padding=padw, bias=use_bias)),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            spectral_norm(nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=(1,1,1), padding=padw, bias=use_bias)),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [spectral_norm(nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=(1,1,1), padding=padw))]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input, mask=None):
        """Standard forward."""
        if self.pooling:
            return self.model(input).sum(dim=(2,3,4))
        else:
            return self.model(input)