#!/bin/bash

# Created by raouf at 19/01/2020

# FaceForensics
echo '-------------------------------'
echo 'getting FaceForensics++ dataset'
echo '-------------------------------'
echo ''

# get the absolute path of the curreent script to make it always work
SCRIPTPATH="$( cd "$(dirname "$0")" || exit ; pwd -P )"

TARGET_DIR=${SCRIPTPATH%/Unsupervised-Spatiotemporal-Data-Inpainting/*}/datasets

if [[ "$1" == "from_scratch" ]]
then
    TARGET_DIR=$TARGET_DIR/FaceForensics

    mkdir -p ${TARGET_DIR}/original-data

    SCRIPT_NAME="download_FaceForensics.py"
    python  $SCRIPTPATH/$SCRIPT_NAME $TARGET_DIR/original-data -d original -c c23 -t videos  --server EU2 --num_videos 10
    mv  -f ${TARGET_DIR}/original-data/original_sequences/youtube/c23/videos/* ${TARGET_DIR}
    rm -rf ${TARGET_DIR}/original-data


    mkdir -p ${TARGET_DIR}/tmp

    python $SCRIPTPATH/preprocess_FaceForensics.py  --dataset_path=$TARGET_DIR

    rm -rf ${TARGET_DIR}/*.mp4
    mv  -f   ${TARGET_DIR}/tmp/* ${TARGET_DIR}
    rm -rf ${TARGET_DIR}/tmp

else
    wget -nc "https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/21339804/FaceForensics.zip"
    unzip -qq "FaceForensics.zip" -d "${TARGET_DIR}/"
    rm -f "FaceForensics.zip"
fi

echo '----------------------'
echo 'FaceForensics++ DONE !'
echo '----------------------'
echo ''
