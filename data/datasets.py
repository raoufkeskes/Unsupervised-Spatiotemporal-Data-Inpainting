import os

from sklearn.model_selection import train_test_split

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from Data.src.occlusions import *

class MotherDataset(Dataset):
    def __init__(self, root_dir, transform=lambda x : x , occlusions=None , nb_frames=15 ):

        self.occlusions     = occlusions
        self.root_dir       = root_dir
        self.transform      = transform
        self.nb_frames      = nb_frames
        self.filenames      = os.listdir(root_dir)

    def __getitem__(self, idx):
        # complete video
        Y = self.read_video(self.filenames[idx])

        # select one occlusion
        occlusion_idx = np.random.randint(low=0, high=len(self.occlusions))
        # occluded video
        X = self.occlusions[occlusion_idx](Y)

        return X, Y

    def __len__(self):
        return len(self.filenames)

    def read_video(self, filename):
        """
        :param filename: String path to video
        :return: raw video Tensor ( Time , Channels , Width , Height )   Ex ( 100 , 3 , 64 , 64 )
        """

        # Open the input movie file
        input_movie = cv2.VideoCapture(self.root_dir + filename)
        length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = input_movie.get(cv2.CAP_PROP_FPS)
        frame_number = 0

        # iterate over frames
        raw_video = torch.Tensor()
        while True:
            # Grab a single frame of video
            ret, frame = input_movie.read()
            frame_number += 1

            # Quit when the input video file ends
            if not ret:
                break

            # Convert the image from BGR color (which OpenCV uses) to RGB color
            # Reorder to Channels x Width x Height
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.transpose(frame, (2, 0, 1))
            frame = torch.tensor(frame)

            if ( self.transform) :
                frame = self.transform(frame)

            # add to video tensor
            raw_video = torch.cat((raw_video,frame.unsqueeze(0)), dim=0)

        # select a starting index to extract a slice
        start_idx = np.random.randint( low=0, high=len(raw_video)-self.nb_frames )

        return raw_video[start_idx:start_idx+self.nb_frames]

class FaceForensics_Dataset(MotherDataset):
    pass

class Bair(MotherDataset):
    pass

class SST(MotherDataset):
    pass

class KTH(MotherDataset):
    pass

def getDataloaders( root="../../../FaceForensics" , batch_size = 4 ):

    test_size = 50
    if "FaceForensics" in root:
        train_dataset = SintelDataset(root, frames_transforms, frames_aug_transforms, co_aug_transforms)
        val_dataset = SintelDataset(root, frames_transforms)
        val_size = 133
    elif "BAIR" in root:
        train_dataset = OceanData(root, frames_transforms, frames_aug_transforms, co_aug_transforms)
        val_dataset = OceanData(root, frames_transforms)
        val_size = 100
    elif 

    torch.manual_seed(1)
    indices = torch.randperm(len(train_dataset)).tolist()
    train_idx, valid_idx, test_idx = indices[val_size + test_size:], indices[:val_size], indices[
                                                                                         val_size:test_size + val_size]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                              pin_memory=torch.cuda.is_available(), num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler,
                            pin_memory=torch.cuda.is_available(), num_workers=4)
    test_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=test_sampler,
                             pin_memory=torch.cuda.is_available(), num_workers=4)

    return train_loader, val_loader, test_loader

def getDataloaders():
# return train test val dataloaders
    pass