import torch
from torch.nn import init

"""
@author : Aissam Djahnine
@date : 17/01/2020 02:39
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


def init_net(net, mode, init_gain=0.02, gpu_ids=0):

    """Initialize a network:

    1. register CPU/GPU device (with multi-GPU support);
    2. initialize the network weights

    Parameters:
        net (network)      -- the network to be initialized
        mode(str)          -- The Batch/conv dimension
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        #net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs

    init_weights(net, mode, init_gain=init_gain)

    return net



