import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from models.utils import *

"""
@author : Aissam Djahnine
@date : 18/01/2020 01:27
Inspired by CycleGan,Pix2Pix paper : https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix 
"""

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def define_D(mode, input_nc, ndf, init_gain=0.02, gpu_ids=[device]):
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
    return init_net(net, mode, init_gain, gpu_ids)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, mode='2', input_nc=3, ndf=32, use_bias=False):
        """Construct a PatchGAN discriminator

        Parameters:
            mode (str)      -- BatchNorm/conv dimension
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            use_bias (bool)     -- if the conv layer uses bias or not
        """
        super(NLayerDiscriminator, self).__init__()

        stride = 2 if mode =='2' else (1,2,2)

        model = [spectral_norm(eval("nn.Conv"+mode+"d")(input_nc, ndf, kernel_size=3, stride=stride, padding=1)),
                 eval("nn.BatchNorm"+mode+"d")(ndf),
                 nn.LeakyReLU(0.2, True)
                ]

        map = 1
        map_update = 1

        for n in range(1,4):
            map_update = map
            map = min(2 ** n, 8)
            model += [
                spectral_norm(eval("nn.Conv"+mode+"d")(ndf * map_update, ndf * map, kernel_size=3, stride=stride, padding=1, bias=use_bias)),
                eval("nn.BatchNorm"+mode+"d")(ndf * map),
                nn.LeakyReLU(0.2, True)
            ]

        for n in range(2):
            model += [
                spectral_norm(eval("nn.Conv"+mode+"d")(ndf * map, ndf * map, kernel_size=3, stride=stride, padding=1, bias=use_bias)),
                eval("nn.BatchNorm"+mode+"d")(ndf * map),
                nn.LeakyReLU(0.2, True)
                 ]

        model += [spectral_norm(eval("nn.Conv"+mode+"d")(ndf * map, ndf * map, kernel_size=3, stride=1, padding=1))]
        model += [spectral_norm(eval("nn.Conv"+mode+"d")(ndf * map, 1, kernel_size=3, stride=1, padding=1))]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


if __name__ == '__main__':

    # test Discriminator :

    # Define input for df :
    input_df = torch.rand((2, 3, 64, 64)).to(device)     # input = Batch_size , channels, width , height
    print(' The input shape for df is : {}'.format(input_df.shape))

    # Define input for ds :
    input_ds = torch.rand((2, 3, 35, 64, 64)).to(device)     # input = Batch_size , channels, frames, width , height
    print(' The input shape for ds is : {}'.format(input_ds.shape))

    # create instance of Discriminator(ds,df) with define_D, set the mode to '2','3', ndf = 32 :
    netD2 = define_D('2', 3, 64)  #df
    netD3 = define_D('3', 3, 64)  #ds

    # check whether the model is on GPU , this function returns a boolean :
    print(' The model --mode : {} is on GPU : {}'.format(2, next(netD2.parameters()).is_cuda))
    print(' The model --mode : {} is on GPU : {}'.format(3, next(netD3.parameters()).is_cuda))

    # Compute the output :
    output_df = netD2(input_df)
    output_ds = netD3(input_ds)

    # check the output of netD2 , netD3 : output=[batch_size, 1 , 1 , 1]
    print(' The output shape for df is : {}'.format(output_df.shape))
    print(' The output shape for ds is : {}'.format(output_ds.shape))

    # calculate number of parameters for df,ds :
    print('Number of Parameters for Df is : {}'.format(sum(p.numel() for p in netD2.parameters())))
    print('Number of Parameters for Ds is : {}'.format(sum(p.numel() for p in netD3.parameters())))



