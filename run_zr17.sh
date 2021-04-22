#!/usr/bin/env bash


FILELIST=$1
SGE=$2
ZRROOT=$3

source ~/miniconda3/etc/profile.d/conda.sh

cd $ZRROOT
conda activate zerospeech

./run_disc $SGE $FILELIST

conda deactivate
