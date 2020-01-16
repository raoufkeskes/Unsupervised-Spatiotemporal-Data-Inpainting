#!/bin/bash

# BAIR
TARGET_DIR=./datasets/BAIR/raw-data
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

tar -xvf ${FILENAME} --strip-components=1 -C "${TARGET_DIR}"
#rm -f ${FILENAME}

echo '------------'
echo 'BAIR  DONE !'
echo '------------'
echo ''