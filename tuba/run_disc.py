import argparse
import sys
from os.path import join
from os import chdir
import subprocess


if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('-s', '--sge', type=str, default='nosge')
	parser.add_argument('-l', '--filelist', type=str, default='')
	parser.add_argument('-zr_root', '--zr_root', type=str, default='/home/korhan/Desktop/zerospeech2017/track2/src/ZRTools')

	args = parser.parse_args()


	chdir(args.zr_root)

	command = './run_disc {} {}'.format(args.sge,args.filelist)

	print(command)

	subprocess.call(command.split())

