#!/usr/bin/env bash


EXPROOT=$4
DTW=$3
EXPNAME=$2
ZRROOT=$1

source ~/miniconda3/etc/profile.d/conda.sh

cd $ZRROOT
conda activate zerospeech

./post_disc $EXPNAME $DTW $EXPROOT

conda deactivate
