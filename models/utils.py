import torch
from torch.nn import init
from models.Discriminator import *

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


def init_net(net, mode, init_gain=0.02, gpu_ids=[0]):

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
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs

    init_weights(net, mode, init_gain=init_gain)

    return net


#####   SORRY GUYS

# Created by raouf at 20/01/2020


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def loss_D( real , fake , netD_seq , netD_frame , occlusion_obj=None ) :
    """
    :param real     : occluded video clip Tensor with shape ( batch_size x Time x C x H x W )
    :param fake     : generated occluded video clip Tensor with shape ( batch_size x Time x C x H x W )
    :return         : GANs loss defined on the paper
    """



    ############################## DISCRIMINATOR SEQUENCE  DS Loss ################################
    DS_criterion = torch.nn.BCEWithLogitsLoss(  )
    DS_predicted_label_real = netD_seq(real).view(-1)
    DS_predicted_label_fake = netD_seq(fake).view(-1)
    DS_predicted_labels     = torch.cat((DS_predicted_label_real,DS_predicted_label_fake),dim=0)

    DS_true_label_fake = torch.ones_like(DS_predicted_label_real) # ones
    DS_true_label_real = torch.zeros_like(DS_predicted_label_fake) # zeros
    DS_true_labels     = torch.cat((DS_true_label_fake,DS_true_label_real),dim=0)

    #DS loss
    DS_loss = DS_criterion(DS_predicted_labels,DS_true_labels)
    print("DS Loss : ",DS_loss)
    ################################################################################################

    ################################ DISCRIMINATOR SEQUENCE  DS ####################################
    DF_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

    DS_predicted_label_real = netD_seq( real.view() ).view( -1 )
    DS_predicted_label_fake = netD_seq( fake ).view( -1 )

    ################################################################################################




if __name__ == '__main__':

    # Define input for ds :
    real = torch.rand((2, 3, 35, 64, 64)).to(device)
    fake = torch.rand((2, 3, 35, 64, 64)).to(device)

    # create instance of Discriminator(ds,df) with define_D, set the mode to '2','3', ndf = 32 :
    netD2 = define_D('2', 3, 64)  #df
    netD3 = define_D('3', 3, 64)  #ds


    loss(real,fake,netD3,netD2)



