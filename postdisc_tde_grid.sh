#!/usr/bin/env bash




if [ $# -ne 4 ]; then
    echo "usage: <filepath> <dtw_th_on> <dtw_th_off> <step>"
    exit 1
fi

filepath=$1
DTW_ON=$2
DTW_OFF=$3
DTWSTEP=$4


while IFS= read -r line
do
	printf '%s\n' "$line"
	./run_post_disc_n_tde.sh $line $DTW_ON $DTW_OFF $DTWSTEP

done <"$filepath"