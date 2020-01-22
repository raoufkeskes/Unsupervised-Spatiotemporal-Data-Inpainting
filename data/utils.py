# Created by raouf at 19/01/2020

from torch.utils.data import DataLoader, SubsetRandomSampler
from data.datasets   import FaceForensics_Dataset , KTH_Dataset
import torch
import cv2
import os
import numpy as np


def getDataloaders(root, transform, occlusions=None, nb_frames=35, batch_size=4, val_size=0.05, test_size=0.05):
    """
    :param root        : String => root directory of the dataset
    :param transform   : torchvision.transforms.Compose(transforms) ==> https://pytorch.org/docs/stable/torchvision/transforms.html
    :param occlusions  : list of occlusions objects ==> Ex [raindrops_obj,remove_pixel_obj]
    :param nb_frames   : int  ==> number of frames for videos ( nb_frames x C x H x W )
    :param batch_size  : int  ==> batch size Ex 4
    :param val_size    : float==> in [0,1]
    :param test_size   : float==> in [0,1]

    :return: train_loader, val_loader, test_loader   for the corresponding dataset with the corresponding parameters
    """
    dataset = None
    if "FaceForensics" in root:
        dataset = FaceForensics_Dataset(root, transform, occlusions=occlusions, nb_frames=nb_frames)
    elif "BAIR" in root:
        pass
    elif "KTH" in root :
        dataset = KTH_Dataset(root, transform, occlusions=occlusions, nb_frames=nb_frames)
    elif "SST" in root :
        pass

    # convert val_size , test_size to int
    val_size  = int( len(dataset) * val_size  )
    test_size = int( len(dataset) * test_size )

    # construct dataloaders
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    train_idx, valid_idx, test_idx = indices[val_size + test_size:], indices[:val_size], indices[val_size:test_size + val_size]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler   = SubsetRandomSampler(valid_idx)
    test_sampler  = SubsetRandomSampler(test_idx)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler,
                              pin_memory=torch.cuda.is_available(), num_workers=0)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler,
                            pin_memory=torch.cuda.is_available(), num_workers=0)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler,
                             pin_memory=torch.cuda.is_available(), num_workers=0)

    return train_loader, val_loader, test_loader




def write_video(video_tensor, out_dir, filename, fps = 30.0 ,occlusion_color=255 ):
    """
    :param video_tensor: Tensor => - shape ( nb_frames x C x H x W )
                                   - values should be in 255
                                   - if you made transformations to your tensor video like ( mean , std ) normalization ... you have to unormalize it before saving it
    :param out_dir: String      =>  Ex : "../outputs/FaceForensics/tmp"
    :param filename: String     =>  Ex : "helloworld.mp4"
    :param fps: Float           => video fps
    :param occlusion_color: int => a gray scale number [ 0 .. 225 ]
    -------------
    write a video
    """
    out_size = video_tensor.shape[-2] , video_tensor.shape[-1]

    # mkdir if not exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # Create an output movie file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    movie  = cv2.VideoWriter(out_dir+"/"+filename , fourcc, fps , out_size )

    for frame in video_tensor:

        # numpy => RGB to BGR ==>  W x H x C to C x W x H
        res = (frame.numpy()[::-1]).transpose(1, 2, 0)
        res[res<0] = occlusion_color

        # convert to np.uint8 type
        res = res.astype(np.uint8)
        # write frame
        movie.write(res)



    movie.release()
    cv2.destroyAllWindows()



