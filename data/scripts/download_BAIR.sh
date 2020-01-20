#!/bin/bash

# Created by raouf at 19/01/2020

# BAIR
TARGET_DIR=./datasets/BAIR
mkdir -p ${TARGET_DIR}

FILENAME="bair_robot_pushing_dataset_v0.tar"
DOMAIN='http://rail.eecs.berkeley.edu'
path='/datasets'
BASE_URL="$DOMAIN$path"

echo '---------------------------------'
echo 'getting BAIR robot action dataset'
echo '---------------------------------'
echo ''


#wget -nc -O ${FILENAME} "${BASE_URL}/${FILENAME}"

tar -xvf ${FILENAME} "${TARGET_DIR}"
#rm -f ${FILENAME}
mv ${TARGET_DIR}/softmotion30_44k/* ${TARGET_DIR}


echo '------------'
echo 'BAIR  DONE !'
echo '------------'
echo ''