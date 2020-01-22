#!/bin/bash

# Created by raouf at 19/01/2020

# get the absolute path of the curreent script
# to make it always work
SCRIPTPATH="$( cd "$(dirname "$0")" || exit ; pwd -P )"

# KTH : Human Actions Dataset
TARGET_DIR=$SCRIPTPATH/../../../datasets/KTH

DOMAIN='http://www.nada.kth.se'
path='/cvap/actions'
BASE_URL="$DOMAIN$path"
FILENAMES="walking jogging running boxing handwaving handclapping"

mkdir -p ${TARGET_DIR}

echo '---------------------------------'
echo 'getting KTH human actions dataset'
echo '---------------------------------'
echo ''
for FILENAME in $FILENAMES; do

    echo "##################"
    echo "getting $FILENAME : "
    echo "##################"

    wget -nc -O "${FILENAME}.zip" "${BASE_URL}/${FILENAME}.zip" 
    unzip -qq "${FILENAME}.zip" -d "${TARGET_DIR}/"
    rm -f "${FILENAME}.zip"

done

echo '----------'
echo 'KTH DONE !'
echo '---------'
echo ''