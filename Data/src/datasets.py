import os

from sklearn.model_selection import train_test_split

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from Data.src.occlusions import *


class FaceForensics_Dataset(Dataset):

    def __init__(self, root_dir, transform=lambda x : x , output_size=(64, 64), occlusion=None , save_occlusion=True):

        self.occlusion      = occlusion
        self.output_size    = output_size
        self.root_dir       = root_dir
        self.transform      = transform
        self.filenames      = os.listdir(root_dir + "/raw-data/")
        self.save_occlusion = save_occlusion

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # complete video
        Y = self.read_video(self.filenames[idx])
        # occluded video
        X = self.occlusion.apply(Y)
        return X, Y

    def read_video(self, filename):
        """
        :param filename: String path to video
        :return: raw video Tensor ( Time , Channels , Width , Height )   Ex ( 100 , 3 , 64 , 64 )
        """

        # Open the input movie file
        input_movie = cv2.VideoCapture(self.root_dir + "/raw-data/" + filename)
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
            torch.cat((raw_video,frame.unsqueeze(0)), dim=0)

        return raw_video
