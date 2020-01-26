#!/bin/bash

# Created by raouf at 19/01/2020

echo '---------------------------------'
echo 'getting BAIR robot action dataset'
echo '---------------------------------'
echo ''

# get the absolute path of the curreent script
# to make it always work
SCRIPTPATH="$( cd "$(dirname "$0")" || exit ; pwd -P )"

# BAIR
TARGET_DIR=$SCRIPTPATH/../../../datasets/BAIR
mkdir -p ${TARGET_DIR}

FILENAME="bair_robot_pushing_dataset_v0.tar"
DOMAIN='http://rail.eecs.berkeley.edu'
path='/datasets'
BASE_URL="$DOMAIN$path"


##wget -nc -O ${FILENAME} "${BASE_URL}/${FILENAME}"
#
#echo $FILENAME
#tar -xvf $SCRIPTPATH/${FILENAME}  -C $SCRIPTPATH

##rm -f ${FILENAME}
python $SCRIPTPATH/preprocess_BAIR.py --out_dir="${TARGET_DIR}"

echo '------------'
echo 'BAIR  DONE !'
echo '------------'
echo ''