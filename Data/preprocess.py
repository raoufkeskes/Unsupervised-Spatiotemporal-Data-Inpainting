import argparse
import cv2
import face_recognition
import os
import numpy as np
from occlusion_utils import *



def main (params) :


    np.random.seed(params.random_seed)
    img_size = (64, 64)
    # initial point for  moving bar 
    mb_initial_point = np.array([0.0, 0.0])

    datasets = params.datasets.split(",")
    occlusions = params.occlusions.split(",")
    for dataset in datasets :

        if ( dataset=="SST" ):
            print("Not yet ...")
            exit(1)

        

        # getting starting variables
        for occlusion in occlusions:
            if (occlusion == "moving_bar"):
                #position_ver_bar = np.random.rand(2)
                mb_position = mb_initial_point
                mb_height   = params.mb_height
                mb_width    = params.mb_width
                mb_speed    = params.mb_speed
                mb_direction= int(params.mb_direction)

            elif (occlusion == "remove_pixels"):
                threshold = params.rp_threshold
            elif (occlusion == "raindrops"):
                rd_number    = params.rd_nbr,
                rd_positions = np.random.rand(params.rd_nbr,2) # np.array([[0.1,0.1],[0.1,0.3],[0.1,0.5],[0.1,0.7],[0.1,0.9]])
                rd_width     = params.rd_width  ,
                rd_height    = np.random.rand(params.rd_nbr)  # np.array([0.2,0.2,0.2,0.2,0.2])
                rd_speed     = np.random.rand(params.rd_nbr)  # np.array([0.1,0.2,0.3,0.4,0.5])


        path = "./datasets/"+str(dataset)
        filenames = os.listdir(path+"/raw-data")
        count_file = 1
        for filename in filenames :
            if ( filename.endswith(".mp4") or filename.endswith(".avi") ):

                print(filename)
                face_locations = None
                print("dataset : {} filename : {} {}|{}".format(dataset,filename,count_file,len(filenames)))

                # Open the input movie file
                input_movie = cv2.VideoCapture(path+"/raw-data/"+filename)
                length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = input_movie.get(cv2.CAP_PROP_FPS)

                #Create an output movie files
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                movie = cv2.VideoWriter(path+"/resized-data/"+str(img_size[0])+"_" + filename, fourcc, 30.0, img_size )
                occluded_movie_ver_bar      = cv2.VideoWriter(path+"/occluded-data/"+str(img_size[0])+"_occluded_bar_" + filename, fourcc, 30.0, img_size )
                occluded_movie_raindrops    = cv2.VideoWriter(path+"/occluded-data/"+str(img_size[0]) + "_occluded_rain_" + filename, fourcc, 30.0, img_size)
                occluded_movie_removepixels = cv2.VideoWriter(path+"/occluded-data/"+str(img_size[0]) + "_occluded_noise_" + filename, fourcc, 30.0, img_size)

                # ,random  parameters for occlusions
                frame_number = 0

                while True:
                    # Grab a single frame of video
                    ret, frame = input_movie.read()
                    frame_number += 1

                    # Quit when the input video file ends
                    if not ret:
                        break


                    # Convert the image from BGR color (which OpenCV uses) to RGB color
                    rgb_frame = frame[:, :, ::-1]
                    if (dataset == "FaceForensics"):
                        ### extend for face recognition ###
                        extend = 60
                        # Find all the faces and face encodings in the current frame of video
                        if (face_locations is None):
                            face_locations = face_recognition.face_locations(rgb_frame)
                            # our dataset have only one face and we are also interested by one FaceForensics
                            top, right, bottom, left = face_locations[0]
                        rgb_frame = rgb_frame[top - extend:bottom + extend, left - extend:right + extend]

                    resized = cv2.resize(rgb_frame, img_size )

                    # resized movie
                    movie.write(resized[:, :, ::-1])

                    # speed = np.random.rand(default_number)

                    for occlusion in occlusions :
                        if ( occlusion == "moving_bar" ) :
                            occluded_movie_ver_bar.write(   moving_vertical_bar(resized.copy(),
                                                            position =mb_position,
                                                            width    =mb_width,
                                                            height   =mb_height)  )
                            # next column position = current column position + speed * ( direction )
                            mb_position[1] = (mb_position[1] + (mb_direction) * mb_speed) % 1

                        elif ( occlusion =="remove_pixels") :
                            occluded_movie_removepixels.write(remove_pixel(resized.copy(), threshold))
                        elif (occlusion == "raindrops"):

                            occluded_movie_raindrops.write( raindrops(resized.copy(),
                                                                 number   =rd_number,
                                                                 positions=rd_positions,
                                                                 width    =params.rd_width ,
                                                                 height   =rd_height ) )
                            rd_positions[:,0] += rd_speed






                    if (frame_number % 50 == 0):
                        print("Writing frame {} / {} of file {}|{} ".format(frame_number, length,count_file,len(filenames)))


                #  All done !
                input_movie.release()
                movie.release()
                occluded_movie_ver_bar.release()
                occluded_movie_removepixels.release()
                occluded_movie_raindrops.release()
                cv2.destroyAllWindows()

                count_file += 1



if __name__ == '__main__':

    """
    cmd parameters : a simple API 
    """
    parser = argparse.ArgumentParser()



    parser.add_argument('--datasets'    , default  = "FaceForensics,KTH"                        , type=str  , metavar='DATASETS'  , help='datasets separated by a ","  Ex : FaceForensics,KTH')
    parser.add_argument('--occlusions'  , default  = "moving_bar,raindrops,remove_pixels"       , type=str  , metavar='OCCLUSIONS', help='occlusions separated by a "," Ex : remove_pixels,raindrops')
    parser.add_argument('--random_seed' , default  = 0                                          , type=int  , metavar='SEED'      , help='random seed for reproducibility ')
    parser.add_argument('--mb_direction', default  = "-1"                                       , type=str  , metavar='MB_DIR'    , help='vertical moving bar direction +1 right , -1 left ')
    parser.add_argument('--mb_width'    , default  = 0.2                                        , type=float, metavar='MB_WIDTH'  , help='the bar width  : scaling float (0,1) ')
    parser.add_argument('--mb_height'   , default  = 1.0                                        , type=float, metavar='MB_HEIGHT' , help='the bar height : scaling float (0,1) ')
    parser.add_argument('--mb_speed'    , default  = 1.0/120                                    , type=float, metavar='MB_SPEED'  , help='the bar moving speed : scaling float (0,1) ')
    parser.add_argument('--rd_nbr'      , default  = np.random.randint(64)                      , type=int  , metavar='RD_NBR'    , help='rain starting positions number ')
    parser.add_argument('--rd_width'    , default  = 2.0/64                                     , type=float, metavar='RD_WIDTH'  , help='rain drops fixed width  : scaling float (0,1)  ')
    parser.add_argument('--rp_threshold', default  = 0.05                                       , type=int  , metavar='RD_HEIGHT' , help='remove pixels fixed threshold : scaling float (0,1)  ')

    args = parser.parse_args()
    main(args)