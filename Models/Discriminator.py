import torch
import torch.nn as nn
from torch.nn import init
"""
@author : Aissam Djahnine
@date : 09/01/2020 20:03
Inspired by CycleGan,Pix2Pix paper : https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix 
"""

def init_weights(net, mode, init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)        -- network to be initialized
        mode (str)           -- BatchNorm/Conv dimension
        init_gain (float)    -- scaling factor for normal
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.normal_(m.weight.data, 0.0, init_gain)

        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)

        elif classname.find('BatchNorm'+mode+'d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, mode, init_gain=0.02, gpu_ids=[0]):

    """Initialize a network:

    1. register CPU/GPU device (with multi-GPU support);
    2. initialize the network weights

    Parameters:
        mode(str)          -- The Batch/conv dimension
        net (network)      -- the network to be initialized
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, mode, init_gain=init_gain)

    return net


def define_D(mode, input_nc, ndf, init_gain=0.02, gpu_ids=[0]):
    """Create a 'PatchGAN' discriminator

    Parameters:
        mode (str)         -- BatchNorm/conv dimension
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator
    """
    net = NLayerDiscriminator(mode, input_nc, ndf)

    return init_net(net,mode, init_gain, gpu_ids)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, mode='2', input_nc=3, ndf=32, use_bias=False):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()

        model = [eval("nn.Conv"+mode+"d")(input_nc, ndf, kernel_size=3, stride=2, padding=1),
                 eval("nn.BatchNorm"+mode+"d")(ndf),
                 nn.LeakyReLU(0.2, True)
                ]

        map = 1
        map_update = 1

        for n in range(1, 3):
            map_update = map
            map = min(2 ** n, 8)
            model += [
                eval("nn.Conv"+mode+"d")(ndf * map_update, ndf * map, kernel_size=3, stride=2, padding=1, bias=use_bias),
                eval("nn.BatchNorm"+mode+"d")(ndf * map),
                nn.LeakyReLU(0.2, True)
            ]

        for n in range(2):
            model += [
                eval("nn.Conv"+mode+"d")(ndf * map, ndf * map, kernel_size=3, stride=1, padding=1, bias=use_bias),
                eval("nn.BatchNorm"+mode+"d")(ndf * map),
                nn.LeakyReLU(0.2, True)
                 ]

        model += [eval("nn.Conv"+mode+"d")(ndf * map, 1, kernel_size=3, stride=1, padding=1)]  # output 1 channel prediction map
        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)