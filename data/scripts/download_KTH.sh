#!/bin/bash


# KTH : Human Actions Dataset

TARGET_DIR=./datasets/KTH
DOMAIN='http://www.nada.kth.se'
path='/cvap/actions'
BASE_URL="$DOMAIN$path"
FILENAMES="walking jogging running boxing handwaving handclapping"

mkdir -p ${TARGET_DIR} ${TARGET_DIR}/raw-data ${TARGET_DIR}/occluded-data ${TARGET_DIR}/resized-data

echo '---------------------------------'
echo 'getting KTH human actions dataset'
echo '---------------------------------'
echo ''
for FILENAME in $FILENAMES; do

    echo "##################"
    echo "getting $FILENAME : "
    echo "##################"

    wget -nc -O "${FILENAME}.zip" "${BASE_URL}/${FILENAME}.zip" 
    unzip -qq "${FILENAME}.zip" -d "${TARGET_DIR}/raw-data/"
    rm -f "${FILENAME}.zip"

done

echo '----------'
echo 'KTH DONE !'
echo '---------'
echo ''