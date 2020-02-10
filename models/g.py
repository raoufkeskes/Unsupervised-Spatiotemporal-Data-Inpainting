import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from models.utils import *


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
    net = ResnetGenerator3d(input_nc, output_nc, ngf,norm_layer)

    return init_net(net, mode, init_gain,gpu_ids)

class Block3d(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, sn=True):

        super().__init__()

        self.b1 = nn.BatchNorm3d(num_features=in_channels)
        self.b2 = nn.BatchNorm3d(num_features=out_channels)
        self.activation = nn.ReLU()
        self.wrapper = spectral_norm if sn else lambda x: x

        self.c1 = self.wrapper(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias))
        self.c2 = self.wrapper(nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=bias))
        self.c_sc = self.wrapper(nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0, bias=bias)) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        h = self.b1(x)
        h = self.activation(h)
        h = self.c1(h)
        h = self.b2(h)
        h = self.activation(h)
        h = self.c2(h)
        x = self.c_sc(x)
        return h + x

class SelfAttention(nn.Module):

    def __init__(self, in_dim, sn=True):
        super().__init__()
        self.chanel_in = in_dim
        self.wrapper = spectral_norm if sn else lambda x: x

        self.conv_theta = self.wrapper(
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1))
        self.conv_phi   = self.wrapper(
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1))
        self.conv_g     = self.wrapper(
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1))
        self.conv_attn  = self.wrapper(
            nn.Conv2d(in_channels=in_dim // 2, out_channels=in_dim, kernel_size=1))

        self.sigma = nn.Parameter(torch.zeros(1))

        self.pool = nn.MaxPool2d((2, 2), stride=2)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """

        mode_3d = x.dim() == 5
        
        if mode_3d:
            batch_size_seq, nc, t, h, w = x.shape
            x = x.view(-1, nc, h, w)

        batch_size, nc, h, w = x.size()
        location_num = h * w
        downsampled_num = location_num // 4
        theta = self.conv_theta(x)

        theta = theta.view(batch_size, nc // 8, location_num).permute(0, 2, 1)

        phi = self.conv_phi(x)

        phi = self.pool(phi)
        phi = phi.view(batch_size, nc // 8, downsampled_num)

        attn = torch.bmm(theta, phi)
        attn = self.softmax(attn)

        g = self.conv_g(x)

        g = self.pool(g)
        g = g.view(batch_size, nc // 2, downsampled_num).permute(0, 2, 1)

        attn_g = torch.bmm(attn, g).permute(0, 2, 1)
        attn_g = attn_g.view(batch_size, nc // 2, w, h)

        attn_g = self.conv_attn(attn_g)
        out = x + self.sigma * attn_g

        if mode_3d:
            out = out.view(batch_size_seq, nc, t, h, w)

        return  out

class ResnetGenerator3d(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=32, use_bias=True):
        super().__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc

        self.blocks_3d = nn.Sequential(
            Block3d(input_nc, ngf),
            Block3d(ngf, ngf * 16),
            Block3d(ngf * 16, ngf * 8), 
            Block3d(ngf * 8, ngf * 4),
            Block3d(ngf * 4, ngf * 2),
            SelfAttention(ngf * 2),
            Block3d(ngf * 2, ngf),
            nn.BatchNorm3d(ngf),
            nn.ReLU(),
            spectral_norm(nn.Conv3d(ngf, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)),
            nn.Tanh(),
        )

    def forward(self, y):
        mask = (y >= 0).int().to(device)
        mask_bar = (y < 0).int().to(device)
        return self.blocks_3d(y) * mask_bar + y * mask