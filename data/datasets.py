# Created by raouf at 19/01/2020


# data processing
from torch.utils.data import Dataset
import torch
import numpy as np
import cv2

# utils
import os

class MotherDataset(Dataset):
    """
    class regrouping common attributes and operations between different datasets
    """
    def __init__(self, root_dir, transform , occlusions , nb_frames ):

        self.occlusions     = occlusions
        self.root_dir       = root_dir
        self.transform      = transform
        self.nb_frames      = nb_frames
        self.filenames      = []

    def __getitem__(self, idx):
        """
        :param idx: index of an element
        :return: an element = Triplet
         ( X : Tensor(nb_frames x C x H x W) , complete video clip
           Y : Tensor(nb_frames x C x H x W) ,  occluded video clip , Tensor(nb_frames x C x H x W)
           i : Int                           ,  index of the selected occlusion among self.occlusions for the current video clip Y
         )
        """

        # complete video Tensor
        X = self.read_video(self.filenames[idx])
        # select one occlusion type
        occlusion_idx = np.random.randint(low=0, high=len(self.occlusions))
        # occluded video Tensor

        Y = self.occlusions[occlusion_idx]( X )


        return X , Y , occlusion_idx

    def __len__(self):
        """
        :return: return dataset length ( total number of videos )
        """
        return len(self.filenames)

    def read_video(self, filename):
        """
        :param filename: String filename  ( NOT THE COMPLETE PATH ! )
        :return: video clip Tensor(nb_frames x C x H x W)
            - the  extracted clip starting index is selected RANDOMLY from the whole video  Ex  Video : 300 frames , extracted video clip : 35 frames , starting index : 40
        """
        # Open the input movie file
        input_movie = cv2.VideoCapture(self.root_dir + filename)
        frame_number = 0
        # iterate over frames
        raw_video = []
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

            if ( self.transform) :
                # default  ToTensor
                frame = self.transform(frame)

            # add to video tensor
            raw_video.append(frame.unsqueeze(0))

        # concatenate all frames
        raw_video = torch.cat(raw_video)

        # select a starting index to extract a slice
        start_idx = np.random.randint( low=0, high=raw_video.shape[0]-self.nb_frames )

        return raw_video[start_idx:start_idx+self.nb_frames]



class FaceForensics_Dataset(MotherDataset):
    """
    FaceForensics Dataset class
    """
    def __init__(self, root_dir, transform , occlusions=None , nb_frames=35  ):
        super().__init__( root_dir, transform , occlusions=occlusions , nb_frames=nb_frames )
        self.filenames = os.listdir(root_dir)


