import argparse
import cv2
import face_recognition
import os


face_locations=None

# Configuration
config = get_config()
root = config["global"]["root"]

################################   global variables  ########################################
# FaceForensics
extend         = config["FaceForensics"]['extend']

def FaceRecognition(rgb_frame , output_size =(64,64) ):


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

    output_size = config["global"]["output_width"] , config["global"]["output_height"]


    path =  root +"/Data/datasets/"+ str(dataset)
    filenames = os.listdir(path + "/original-data")
    count_file = 0
    for filename in filenames:
        # if (filename.endswith(".mp4")):
        if ( filename =="720.mp4" ):
            count_file += 1

            # Open the input movie file
            input_movie = cv2.VideoCapture(path + "/original-data/" + filename)
            fps = input_movie.get(cv2.CAP_PROP_FPS)

            # Create an output movie files
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_movie = cv2.VideoWriter(path + "/raw-data/"+filename, fourcc, fps , output_size)

            print(" Preprocessing {} ==> {} : video {}|{} ".format(dataset, filename, count_file, len(filenames)))

            frame_number = 0
            while True:
                # Grab a single frame of video
                ret, frame = input_movie.read()
                frame_number += 1

                # Quit when the input video file ends
                if not ret:
                    break
                if ( dataset=="FaceForensics") :
                    img  = FaceRecognition ( frame[:,:,::-1]  , output_size )
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

    args = parser.parse_args()
    main(args)
