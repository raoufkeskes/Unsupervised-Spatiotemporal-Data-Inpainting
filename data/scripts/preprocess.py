# Created by raouf at 19/01/2020

"""

a script dedicated to preprocess datasets
for example
    - FaceForensics_Dataset ==> we extract faces from raw videos

"""

import argparse
import cv2
import face_recognition
import os


face_locations=None

def FaceRecognition(rgb_frame , output_size =(64,64) , extend=60 ):
    """
    :param rgb_frame: the frame in RGB format (W , H , 3)
    :param output_size: the size of the output frame
    :param extend: number of pixels to extend while extracting the faces from the original video ( the extracted boxes are very focused on the face so we extend it a little bit )
    #------#
    :return: an rgb resized frame with the corresponding face extracted from the original frame
    """
    global face_locations

    # our dataset have only one face and we are also interested by one FaceForensics
    if (face_locations is None):
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_frame)
    top, right, bottom, left = face_locations[0]
    # min max to not overlap
    rgb_frame = rgb_frame[ max(top-extend,0) : min( bottom+extend , rgb_frame.shape[0] ) , max(left-extend,0) : min(right+extend,rgb_frame.shape[1]) ]

    resized = cv2.resize(rgb_frame, output_size)

    return resized

def main(args):

    global face_locations
    dataset = args.dataset
    extend  = args.extend
    output_size =(64,64)

    root = "../datasets/"
    path = root + str(dataset)
    filenames = os.listdir(path)

    count_file = 0
    for filename in filenames:
        if (filename.endswith(".mp4")):
            count_file += 1

            # Open the input movie file
            input_movie = cv2.VideoCapture(path +"/"+ filename)
            fps = input_movie.get(cv2.CAP_PROP_FPS)

            # Create an output movie files
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_movie = cv2.VideoWriter(path + "/tmp/"+filename, fourcc, fps , output_size)

            print(" Preprocessing {} ==> {} : video {}|{} ".format(dataset, filename, count_file, len(filenames)-1 ))

            frame_number = 0
            while True:
                # Grab a single frame of video
                ret, frame = input_movie.read()
                frame_number += 1

                # Quit when the input video file ends
                if not ret:
                    break
                if ( dataset=="FaceForensics") :
                    img  = FaceRecognition ( frame[:,:,::-1]  , output_size ,  extend=extend )
                else :
                    img = "TO DO"


                output_movie.write(img[:, :, ::-1])

            #  All frame done !
            input_movie.release()
            output_movie.release()
            cv2.destroyAllWindows()

            face_locations=None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default="FaceForensics", type=str, metavar='DATASET',
                        help='dataset FaceForensics or KTH or SST or BAIR'  )
    parser.add_argument('--extend', default=60 , type=int, metavar='EXTEND',
                        help='number of pixels to extend while extracting the faces from the original video')

    args = parser.parse_args()
    main(args)
