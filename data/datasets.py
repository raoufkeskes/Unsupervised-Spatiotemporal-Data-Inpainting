# Created by raouf at 19/01/2020


# data processing
from torch.utils.data import Dataset
import torch
import numpy as np
import cv2
from PIL import Image


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

    def read_video(self, filename ):
        """
        :param filename: String filename  ( NOT THE COMPLETE PATH ! )
        :return: video clip Tensor(nb_frames x C x H x W)
            - the  extracted clip starting index is selected RANDOMLY from the whole video  Ex  Video : 300 frames , extracted video clip : 35 frames , starting index : 40
        """
        # Open the input movie file
        input_movie = cv2.VideoCapture(self.root_dir + filename)
        total_nbr_frames = int( input_movie.get( cv2.CAP_PROP_FRAME_COUNT ) )
        # select a starting index to extract a slice
        start_idx = np.random.randint( low=0, high=total_nbr_frames - self.nb_frames )
        # iterate over frames
        raw_video = []

        nbr_read_frames = 0
        while True:

            # Grab a single frame of video
            ret, frame = input_movie.read()
            nbr_read_frames += 1

            # Quit when the input video file ends
            if not ret:
                break

            # we did not seek the starting index for our clip
            if start_idx > (nbr_read_frames-1) :
                continue
            # if it is the end of the video clip slice  stop reading
            if ( (start_idx + self.nb_frames+1 ) == nbr_read_frames ):
                break

            # Convert the image from BGR color (which OpenCV uses) to RGB color
            # Reorder to Channels x Width x Height
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if (self.transform) :
                frame = self.transform(Image.fromarray(frame,"RGB"))

            # add to video tensor
            raw_video.append(frame.unsqueeze(0))
        # concatenate all frames
        raw_video = torch.cat(raw_video)


        return raw_video


class FaceForensics_Dataset(MotherDataset):
    """
    FaceForensics Dataset class
    """
    def __init__(self, root_dir, transform , occlusions=None , nb_frames=35  ):
        super().__init__( root_dir, transform , occlusions=occlusions , nb_frames=nb_frames )
        self.filenames = os.listdir(root_dir)


class KTH_Dataset(MotherDataset):
    """
    FaceForensics Dataset class
    """
    def __init__(self, root_dir, transform , occlusions=None , nb_frames=35  ):
        super().__init__( root_dir, transform , occlusions=occlusions , nb_frames=nb_frames )
        self.filenames = os.listdir(root_dir)


