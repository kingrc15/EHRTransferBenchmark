#!/bin/bash
set -e
export ABSOLUTE_PATH_TO_SPLITS="/data/datasets/eICU2MIMIC/new_split/"
export ABSOLUTE_PATH_TO_ROOT="${ABSOLUTE_PATH_TO_SPLITS}root/"

cd $ABSOLUTE_PATH_TO_SPLITS

s=("phenotyping_split/"
 "in-hospital-mortality_split/" 
 "length-of-stay_split/"
 "decompensation_split/"
 "phenotyping/"
 "in-hospital-mortality/" 
 "length-of-stay/"
 "decompensation/")

for n in ${s[@]};
do
    cd $n
    ln -s $ABSOLUTE_PATH_TO_ROOT train
    ln -s $ABSOLUTE_PATH_TO_ROOT test
    cd ..
done

echo "DONE!"