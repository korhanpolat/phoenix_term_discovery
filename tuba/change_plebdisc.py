import argparse
import sys
from os.path import join



if __name__ == '__main__':


	D=10
	R=7
	castthr=7
	trimthr=0.25 

	parser = argparse.ArgumentParser()

	parser.add_argument('-rhothr', '--rhothr', type=float, default=0.2)
	parser.add_argument('-T', '--T', type=float, default=0.5)
	parser.add_argument('-dx', '--dx', type=int, default=10)
	parser.add_argument('-medthr', '--medthr', type=float, default=0.6)
	parser.add_argument('-Tscore', '--Tscore', type=float, default=0.5)
	parser.add_argument('-zr_root', '--zr_root', type=str, default='/home/korhan/Desktop/zerospeech2017/track2/src/ZRTools')

	args = parser.parse_args()



	with open(join(args.zr_root, 'scripts/plebdisc_filepair')) as f:
	    lines = f.readlines()

	assert lines[29][:17] == 'plebdisc/plebdisc'

	# remove(join(zr_root, 'scripts/plebdisc_filepair'))

	suffix = ' -file1 $LSH1 -file2 $LSH2 | awk \'NF == 2 || $5 > 0. {print $0;}\' > $TMP/${BASE1}_$BASE2.match\n'

	new_cmd = 'plebdisc/plebdisc -S 64 -P 8 -rhothr {} -T {} -B 50 -D {} -dtwscore 1 \
	            -kws 0 -dx {} -medthr {} -twopass 1 -maxframes 90000 -Tscore {} -R {} -castthr {} -trimthr {}'.format(
	            args.rhothr, args.T, D, args.dx, args.medthr, args.Tscore, R, castthr, trimthr)

	lines[29] = new_cmd + suffix

	with open(join(args.zr_root, 'scripts/plebdisc_filepair'), 'w') as file:
	    file.writelines(lines)


