

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.parallel
from utils import *

"""
@author : Aissam Djahnine
@date : 18/01/2020 01:27
Inspired by CycleGan,Pix2Pix paper : https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix 
"""

device = torch.device( 'cuda' ) if torch.cuda.is_available() else torch.device( 'cpu' )


class Block3d( nn.Module ):
    def __init__(self, in_channels, out_channels, bias=False, sn=True):
        super().__init__()

        self.b1 = nn.BatchNorm3d( num_features=in_channels )
        self.b2 = nn.BatchNorm3d( num_features=out_channels )
        self.activation = nn.ReLU()
        self.wrapper = spectral_norm if sn else lambda x: x

        self.c1 = self.wrapper( nn.Conv3d( in_channels, out_channels, kernel_size=3, padding=1, bias=bias ) )
        self.c2 = self.wrapper( nn.Conv3d( out_channels, out_channels, kernel_size=3, padding=1, bias=bias ) )
        self.c_sc = self.wrapper( nn.Conv3d( in_channels, out_channels, kernel_size=1, padding=0,
                                             bias=bias ) ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        h = self.b1( x )
        h = self.activation( h )
        h = self.c1( h )
        h = self.b2( h )
        h = self.activation( h )
        h = self.c2( h )
        x = self.c_sc( x )
        return h + x


class SelfAttention( nn.Module ):

    def __init__(self, in_dim, sn=True):
        super().__init__()
        self.chanel_in = in_dim
        self.wrapper = spectral_norm if sn else lambda x: x

        self.conv_theta = self.wrapper(
            nn.Conv2d( in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1 ) )
        self.conv_phi = self.wrapper(
            nn.Conv2d( in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1 ) )
        self.conv_g = self.wrapper(
            nn.Conv2d( in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1 ) )
        self.conv_attn = self.wrapper(
            nn.Conv2d( in_channels=in_dim // 2, out_channels=in_dim, kernel_size=1 ) )

        self.sigma = nn.Parameter( torch.zeros( 1 ) )

        self.pool = nn.MaxPool2d( (2, 2), stride=2 )

        self.softmax = nn.Softmax( dim=-1 )

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
            x = x.view( -1, nc, h, w )

        batch_size, nc, h, w = x.size()
        location_num = h * w
        downsampled_num = location_num // 4
        theta = self.conv_theta( x )

        theta = theta.view( batch_size, nc // 8, location_num ).permute( 0, 2, 1 )

        phi = self.conv_phi( x )

        phi = self.pool( phi )
        phi = phi.view( batch_size, nc // 8, downsampled_num )

        attn = torch.bmm( theta, phi )
        attn = self.softmax( attn )

        g = self.conv_g( x )

        g = self.pool( g )
        g = g.view( batch_size, nc // 2, downsampled_num ).permute( 0, 2, 1 )

        attn_g = torch.bmm( attn, g ).permute( 0, 2, 1 )
        attn_g = attn_g.view( batch_size, nc // 2, w, h )

        attn_g = self.conv_attn( attn_g )
        out = x + self.sigma * attn_g

        if mode_3d:
            out = out.view( batch_size_seq, nc, t, h, w )

        return out


class ResnetGenerator( nn.Module ):
    def __init__(self, ngf, input_nc=3, output_nc=3, use_bias=True):
        super().__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc

        self.blocks_3d = nn.Sequential(
            Block3d( input_nc, ngf ),
            Block3d( ngf, ngf * 16 ),
            Block3d( ngf * 16, ngf * 8 ),
            Block3d( ngf * 8, ngf * 4 ),
            Block3d( ngf * 4, ngf * 2 ),
            SelfAttention( ngf * 2 ),
            Block3d( ngf * 2, ngf ),
            nn.BatchNorm3d( ngf ),
            nn.ReLU(),
            spectral_norm( nn.Conv3d( ngf, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias ) ),
            nn.Tanh(),
        )

    def forward(self, y, mask=-100 ):
        m = (y >= 0).int()
        m_bar = (y < 0).int()
        x_hat =  self.blocks_3d( y )

        return  x_hat * m_bar  + y * m

if __name__ == '__main__':
    ## test Generator :

    # Define input for generator :
    input_g = torch.rand( (1, 3, 15, 64, 64) ).to( device )  # input =[Batch_size , channels, frames width , height]
    print( ' The input shape is : {}'.format( input_g.shape ) )

    # create instance of generator with define_G , ndf = 32 :
    netG = ResnetGenerator( 64, 3, 3 ).to( device )
    # check whether the model is on GPU , this function returns a boolean :
    print( ' The model is on GPU : {}'.format( next( netG.parameters() ).is_cuda ) )

    # Compute the output :
    output_g = netG( input_g )

    # check the output of netG : output = [Batch_size , channels, frames width , height]
    print( ' The output shape is : {}'.format( output_g.shape ) )

    # calculate number of parameters for generator :

    print( 'Number of Parameters is : {}'.format( sum( p.numel() for p in netG.parameters() ) ) )
