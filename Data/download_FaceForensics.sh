#!/bin/bash

# FaceForensics
echo '-------------------------------'
echo 'getting FaceForensics++ dataset'
echo '-------------------------------'
echo ''

TARGET_DIR=./datasets/FaceForensics
mkdir -p ${TARGET_DIR} ${TARGET_DIR}/raw-data ${TARGET_DIR}/resized-data ${TARGET_DIR}/occluded-data 

SCRIPT_NAME="FaceForensics_download.py" 
python  $SCRIPT_NAME $TARGET_DIR/raw-data -d original -c c23 -t videos --num_videos 10
mv  -f ${TARGET_DIR}/raw-data/original_sequences/youtube/c23/videos/* ${TARGET_DIR}/raw-data
cd ${TARGET_DIR}/raw-data
rm -rf original_sequences
cd ../../

echo '----------------------'
echo 'FaceForensics++ DONE !'
echo '----------------------'
echo ''