import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import spectral_norm
import torch.nn.parallel
from models.utils import *


"""
@author : Aissam Djahnine
@date : 18/01/2020 01:27
Inspired by CycleGan,Pix2Pix paper : https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix 
"""

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def define_G(input_nc, output_nc, ngf, norm=nn.BatchNorm3d, init_gain=0.02, gpu_ids=[device], mode='3'):

    """Create a generator

    Parameters:
        input_nc (int)        -- the number of channels in input images
        output_nc (int)       -- the number of channels in output images
        ngf (int)             -- the number of filters in the last conv layer
        norm (str)            -- the name of normalization layers used in the network: 3D batch
        init_gain (float)     -- scaling factor for normal
        gpu_ids (int list)    -- which GPUs the network runs on: e.g., 0,1,2
        mode (str)            -- BatchNorm/conv dimension

    Returns a generator
    """

    norm_layer = norm
    net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer)

    return init_net(net, mode, init_gain,gpu_ids)


def build_conv_block(dim_in, dim_out, norm_layer, use_bias=False):

    """Construct a convolutional block.

    Parameters:
        dim_in (int)        -- the number of channels_in in the conv layer.
        dim_out (int)       -- the number of channels_out in the conv layer.
        norm_layer          -- normalization layer
        use_bias (bool)     -- if the conv layer uses bias or not

    Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
    """

    conv_block = []

    conv_block += [spectral_norm(nn.Conv3d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=use_bias)),
                   norm_layer(dim_out),
                   nn.ReLU(True)
                   ]

    conv_block += [spectral_norm(nn.Conv3d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=use_bias)),
                   norm_layer(dim_out)
                   ]

    return nn.Sequential(*conv_block)


class ResnetBlock(nn.Module):

    """Define a Resnet block"""

    def __init__(self, dim_in, dim_out, norm_layer, use_bias=False):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections.

        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        """

        super(ResnetBlock, self).__init__()

        self.conv_block = build_conv_block(dim_in, dim_out, norm_layer, use_bias=False)
        self.conv_identity = spectral_norm(nn.Conv3d(dim_in, dim_out, kernel_size=1, padding=0, bias=use_bias))

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = self.conv_identity(x) + self.conv_block(x)  # add skip connections
        return out


class SelfAttention(nn.Module):
    """ Define a Self-attention Layer

    Inspired from SAGAN paper

    """
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = spectral_norm(nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1))  # g(x)
        self.key_conv = spectral_norm(nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1))    # f(x)
        self.value_conv = spectral_norm(nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1))       # h(x)

        # Scale factor
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X T X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (with N = Width*Height)
        """
        batch_size, nc, t, width, height = x.size()
        # reshaping x to [ batch_size* , nc , width , height ]
        x = x.view(-1, nc, width, height)

        m_batch_size,nc, width, height = x.size()

        query = self.query_conv(x).view(m_batch_size, -1, width * height).permute(0, 2, 1)  # B X C X N
        key = self.key_conv(x).view(m_batch_size, -1, width * height)                       # B X C x N
        value = self.value_conv(x).view(m_batch_size, -1, width * height)                   # B X C X N

        energy = torch.bmm(query, key)

        attention = self.softmax(energy)  # B X N X N

        out = torch.bmm(value, attention)

        out = out.view(batch_size, nc, t, width, height)

        out = self.gamma * out + x.view(batch_size, nc, t, width, height)

        return out


class ResnetGenerator(nn.Module):

    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.BatchNorm3d, use_bias=False):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels_in in input images
            output_nc (int)     -- the number of channels_out in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
        """

        super(ResnetGenerator, self).__init__()

        model = []

        # Encoder :
        model += [ResnetBlock(input_nc, ngf, norm_layer=norm_layer, use_bias=use_bias)]
        model += [ResnetBlock(ngf, ngf * 16, norm_layer=norm_layer, use_bias=use_bias)]
        model += [ResnetBlock(ngf * 16, ngf * 16, norm_layer=norm_layer, use_bias=use_bias)]

        # Decoder :
        model += [ResnetBlock(ngf * 16, 8 * ngf, norm_layer=norm_layer, use_bias=use_bias)]
        model += [ResnetBlock(8 * ngf, 4 * ngf, norm_layer=norm_layer, use_bias=use_bias)]
        model += [ResnetBlock(4 * ngf, 2 * ngf, norm_layer=norm_layer, use_bias=use_bias)]

        # Spatial Self attention
        model += [SelfAttention(2 * ngf)]

        model += [ResnetBlock(2 * ngf, ngf, norm_layer=norm_layer, use_bias=use_bias)]
        model += [nn.BatchNorm3d(ngf),
                  nn.ReLU(True)
                  ]

        model += [spectral_norm(nn.Conv3d(ngf, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)),
                  nn.Tanh()
                  ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """<forward>"""
        return self.model(input)


if __name__ == '__main__':
    ## test Generator :

    # Define input for generator :
    input_g = torch.rand((1, 3, 15, 64, 64)).to(device)  # input =[Batch_size , channels, frames width , height]
    print(' The input shape is : {}'.format(input_g.shape))

    # create instance of generator with define_G , ndf = 32 :
    netG = define_G(3, 3, 64)
    # check whether the model is on GPU , this function returns a boolean :
    print(' The model is on GPU : {}'.format(next(netG.parameters()).is_cuda))

    # Compute the output :
    output_g = netG(input_g)

    # check the output of netG : output = [Batch_size , channels, frames width , height]
    print(' The output shape is : {}'.format(output_g.shape))

    # calculate number of parameters for generator :

    print('Number of Parameters is : {}'.format(sum(p.numel() for p in netG.parameters())))


