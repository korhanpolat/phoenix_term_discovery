import argparse
import sys
from os.path import join
from os import chdir, makedirs
import subprocess
from shutil import copyfile, copytree

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-l', '--filelist', type=str, default='')
    parser.add_argument('-th', '--dtw_thr', type=float, default=0.90)

    args = parser.parse_args()


	with open(args.filelist,'r') as f:
		expnames =  [x.strip('\n') for x in f.readlines()]


	for name in expnames:
	    chdir(name)

	    command = './post_disc {} {}'.format(exp_name+'.lsh64', dtw_thr)

	    print(command)

	    subprocess.call(command.split())

		chdir('..')
