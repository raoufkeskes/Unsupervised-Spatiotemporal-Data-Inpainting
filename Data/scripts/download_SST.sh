#!/bin/bash

#SST 

# credentials for SST data
USERNAME="akeskes"
PASSWORD="Amal_Project2020"


TARGET_DIR=./datasets/SST
mkdir -p ${TARGET_DIR} ${TARGET_DIR}/raw-data ${TARGET_DIR}/occluded-data  ${TARGET_DIR}/resized-data

DOMAIN='http://nrt.cmems-du.eu'
path='/motu-web/Motu'
BASE_URL="$DOMAIN$path"


echo '--------------------------------------'
echo 'getting SST dataset (may take a while)'
echo 'Do not worry if you see the message  :'
echo '    "Product is not yet available"    '
echo '       IT WILL WORK JUST WAIT         '
echo '--------------------------------------'
echo ''

echo

echo '##################'
echo 'getting test data'
echo '##################'
echo ''
FILENAME="test.nc"
python3 -m motuclient \
        --motu ${BASE_URL} \
        --service-id GLOBAL_ANALYSIS_FORECAST_PHY_001_024-TDS \
        --product-id global-analysis-forecast-phy-001-024-hourly-t-u-v-ssh \
        --longitude-min -20 \
        --longitude-max -14 \
        --latitude-min -20 \
        --latitude-max -14 \
        --date-min "2019-01-01 00:30:00" \
        --date-max "2019-03-01 23:30:00" \
        --depth-min 0.493 \
        --depth-max 0.4942 \
        --variable thetao \
        --out-dir ${TARGET_DIR}/raw-data \
        --out-name ${FILENAME} \
        --user ${USERNAME} \
        --pwd ${PASSWORD}

echo '##################'
echo 'test data done ! '
echo '##################'
echo ''
#
#echo '##################'
#echo 'getting train data'
#echo '##################'
#echo ''
#FILENAME="train.nc"
#python3 -m motuclient \
#            --motu $BASE_URL  \
#            --service-id GLOBAL_ANALYSIS_FORECAST_PHY_001_024-TDS \
#            --product-id global-analysis-forecast-phy-001-024-hourly-t-u-v-ssh \
#            --longitude-min '-40' \
#            --longitude-max '-34' \
#            --latitude-min '20' \
#            --latitude-max '26' \
#            --date-min "2018-01-01 00:30:00" \
#            --date-max "2018-12-31 23:30:00" \
#            --depth-min 0.493 \
#            --depth-max 0.4942 \
#            --variable thetao \
#            --out-dir ${TARGET_DIR}/raw-data \
#            --out-name ${FILENAME} \
#            --user ${USERNAME} \
#            --pwd  ${PASSWORD}
#echo '##################'
#echo 'train data done !'
#echo '##################'
#echo ''

echo '##################'
echo 'getting clouds   '
echo '##################'
echo ''

wget -nc -O "${TARGET_DIR}/raw-data/cloud_dataset.npy" "https://ndownloader.figshare.com/files/17869082"

echo '##################'
echo 'clouds data done !'
echo '##################'
echo ''
echo ''



echo '-----------'
echo 'SST DONE ! '
echo '-----------'
echo ''