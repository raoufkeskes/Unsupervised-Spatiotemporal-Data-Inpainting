#!/bin/bash

# if no arguments download all the datasets
if [ $# -eq 0 ]
  then
    for script in ./download_*.sh ; do bash $script ; done
  # else download the corresponding dataset
  else
    for dataset in "$@" ; do bash "download_$dataset.sh" ; done
    # for dataset in "$@" ; do bash "test$dataset.sh" ; done
fi
