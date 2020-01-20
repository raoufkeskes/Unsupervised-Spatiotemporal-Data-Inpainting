#!/bin/bash

# Created by raouf at 19/01/2020

# FaceForensics
echo '-------------------------------'
echo 'getting FaceForensics++ dataset'
echo '-------------------------------'
echo ''

TARGET_DIR=../datasets/FaceForensics
mkdir -p ${TARGET_DIR}/original-data

SCRIPT_NAME="FaceForensics_download.py"
python  data/scripts/$SCRIPT_NAME $TARGET_DIR/original-data -d original -c c23 -t videos  --num_videos 10
mv  -f ${TARGET_DIR}/original-data/original_sequences/youtube/c23/videos/* ${TARGET_DIR}
rm -rf ${TARGET_DIR}/original-data


mkdir -p ${TARGET_DIR}/tmp

python data/scripts/preprocess.py  --dataset="FaceForensics"

rm -rf ${TARGET_DIR}/*.mp4
mv  -f   ${TARGET_DIR}/tmp/* ${TARGET_DIR}
rm -rf ${TARGET_DIR}/tmp

echo '----------------------'
echo 'FaceForensics++ DONE !'
echo '----------------------'
echo ''
