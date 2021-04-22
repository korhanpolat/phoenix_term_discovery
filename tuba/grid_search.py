import time
from sklearn.model_selection import ParameterGrid
import argparse
import sys
from os.path import join
from os import chdir, makedirs
import subprocess
from shutil import copyfile, copytree


def run_disc(zr_root, filelist, sge='nosge'):

    # import subprocess

    chdir(zr_root)

    command = './run_disc {} {}'.format(sge, filelist)

    print(command)

    subprocess.call(command.split())

    return command


def post_disc(zr_root, exp_name, dtw_thr):

    chdir(zr_root)

    command = './post_disc {} {}'.format(exp_name+'.lsh64', dtw_thr)

    print(command)

    subprocess.call(command.split())

    results_path = join(zr_root, 'exp', exp_name + '.lsh64', 'results')

    return results_path


def change_plebdisc_thr(zr_root, rhothr=0, T=0.25, D=5, dx=25, medthr=0.5, Tscore=0.5, R=10, castthr=7, trimthr=0.25 ):

    with open(join(zr_root, 'scripts/plebdisc_filepair')) as f:
        lines = f.readlines()

    assert lines[29][:17] == 'plebdisc/plebdisc'

    # remove(join(zr_root, 'scripts/plebdisc_filepair'))

    suffix = ' -file1 $LSH1 -file2 $LSH2 | awk \'NF == 2 || $5 > 0. {print $0;}\' > $TMP/${BASE1}_$BASE2.match\n'

    new_cmd = 'plebdisc/plebdisc -S 64 -P 8 -rhothr {} -T {} -B 50 -D {} -dtwscore 1 \
                -kws 0 -dx {} -medthr {} -twopass 1 -maxframes 90000 -Tscore {} -R {} -castthr {} -trimthr {}'.format(
                rhothr, T, D, dx, medthr, Tscore, R, castthr, trimthr)

    lines[29] = new_cmd + suffix

    with open(join(zr_root, 'scripts/plebdisc_filepair'), 'w') as file:
        file.writelines(lines)




if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('-s', '--sge', type=str, default='nosge')
	parser.add_argument('-l', '--filelist', type=str, default='')
	parser.add_argument('-zr', '--zr_root', type=str, default='/home/korhan/Desktop/zerospeech2017/track2/src/ZRTools')

	args = parser.parse_args()



	param_grid = {  'rhothr': [0.2], 
	              'T':[0.5], 
	              'D': [10], 
	              'dx': [10], 
	              'medthr' : [ 0.5, 0.6], 
	              'Tscore' : [.5], 
	              'R' : [7], 
	              'castthr' : [7], 
	              'trimthr': [0.25]
	                }

	grid = ParameterGrid(param_grid)

	files_list = []


	n_iter = 1
	for vec in grid.param_grid[0].values(): n_iter *= len(vec)

	for i,params in enumerate(grid):

	    print('-------- {} / {} --------'.format(i,n_iter))
	    
	    print(params)

	    file_list_name = args.filelist[:-10] + ''.join('_'.join([str(v) for v in params.values()]).split('.')) + '.lsh64.lst'

	    files_list.append(file_list_name)

	    copyfile(args.filelist, file_list_name)

	    exp_name = file_list_name.split('/')[-1][:-10]

	    try:

	        zr_root = args.zr_root + '_' + exp_name
	        copytree(args.zr_root, zr_root)

	        change_plebdisc_thr(zr_root=zr_root,
	        					rhothr=params['rhothr'], 
	                            T=params['T'], 
	                            D=params['D'], 
	                            dx=params['dx'], 
	                            medthr=params['medthr'], 
	                            Tscore=params['Tscore'], 
	                            R=params['R'], 
	                            castthr=params['castthr'], 
	                            trimthr=params['trimthr'])


	        run_disc(zr_root, file_list_name, args.sge)

	        
	        results_path = post_disc(zr_root, exp_name, 0.93)


	    except Exception as e:
	        print(e.message, e.args)


	with open(join(args.zr_root, 'grid_files.txt'), 'w') as f:
		for line in files_list:
			f.writelines(line + '\n')