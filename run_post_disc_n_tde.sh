#!/usr/bin/env bash


# defaults: zr_exp_root, 

ZRROOT='/home/korhan/Desktop/zerospeech2017/track2/src/ZRTools'
TDEROOT='/home/korhan/Desktop/tez/tdev2/tdev2'
OUTDIR='/home/korhan/Dropbox/tez/exp_results'
CORPUS='phoenix'



if [ $# -ne 4 ]; then
    echo "usage: <exp_name> <dtw_th_on> <dtw_th_range> <step>"
    exit 1
fi

EXPNAME=$1
DTW_ON=$2
DTW_OFF=$3
DTWSTEP=$4


#part1=`dirname "$EXPNAME"`
base_expname=`basename "$EXPNAME"`

if [ ${base_expname: -6} == '.lsh64' ]
then
	base_expname=${base_expname:0:-6}	
fi


for dtw in `seq $DTW_ON $DTWSTEP $DTW_OFF`
do
	echo '====== dtw: ' $dtw ' ======'
	# run post-discovery
	cd $ZRROOT
	source activate zerospeech
	./post_disc $EXPNAME 0.$dtw
	source deactivate

	# run evaluation
	cd $TDEROOT
	source activate TDE
	OUTNAME=$( printf 'dtw_%d.json' $dtw )
	mkdir -p $OUTDIR/$base_expname
	echo $OUTNAME
	python eval_sign.py $ZRROOT/exp/$EXPNAME $CORPUS $OUTDIR/$base_expname/$OUTNAME
	source deactivate
done










