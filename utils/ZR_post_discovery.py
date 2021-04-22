import numpy as np
from os.path import join, isfile
from os import chdir, makedirs, remove
import pandas as pd
import matplotlib.pyplot as plt
from shutil import rmtree, copyfile
import subprocess
import glob
import os



def post_disc(exp_name, dtw_thr, zr_root):
    "run conn-comp clustering"

    chdir(zr_root)

    command = './post_disc {} {}'.format(exp_name+'.lsh64', dtw_thr)

    print(command)

    subprocess.call(command.split())

    results_path = join(zr_root, 'exp', exp_name + '.lsh64', 'results')

    return results_path



def new_match_dict(last_filepair, new_line):
    match = {'f1': last_filepair[0],
             'f2': last_filepair[1],
             'f1_start': int(new_line[0]),
             'f1_end': int(new_line[1]),
             'f2_start': int(new_line[2]),
             'f2_end': int(new_line[3]),
             'score': float(new_line[4])
            }

    if len(new_line) == 6:
        match['rho'] = float(new_line[5])

    return match


def read_matches_outfile(match_file_path):

    min_frame_th = 5

    matches_list = []

    match_file = open(match_file_path, 'r')

    last_filepair = match_file.readline().strip('\n').split(' ')
#    print((last_filepair))

    for i, line in enumerate(match_file):

        new_line = line.strip('\n').split(' ')

        if len(new_line) == 2:
            last_filepair = new_line

        elif len(new_line) == 6:
            if ( (int(new_line[1])-int(new_line[0])) >= min_frame_th ) & ( (int(new_line[3])-int(new_line[2])) >= min_frame_th ):
                matches_list.append(new_match_dict(last_filepair, new_line))

        else:
            print('ERROR: unexpected line: {}'.format(new_line))

    #    if i >9: break
    match_file.close()

    return matches_list


def get_matches_df(match_file_path):
    # type: (str) -> pd.DataFrame

    if 'out' not in match_file_path:     # if exp_name is given
        match_file_path = join(zr_root, 'exp', match_file_path + '.lsh64', 'matches', 'out.1')

    matches_list = read_matches_outfile(match_file_path)

    matches_df = pd.DataFrame.from_records(matches_list,
                                           columns=['f1', 'f2', 'f1_start', 'f1_end', 'f2_start', 'f2_end', 'score', 'rho']
                                           )

    matches_df = matches_df.astype(dtype={'f1': str, 'f2': str,
                                          'f1_start': int, 'f1_end': int, 'f2_start': int, 'f2_end': int,
                                          'score': float, 'rho': float}
                                   )  # type: pd.DataFrame
#    print(matches_df.head(3))
    print('Read {} matches'.format(len(matches_df)))

    return matches_df


def get_matches_all(exp_name, exp_root='/home/korhan/Desktop/zerospeech2017/track2/src/ZRTools/exp/'):

    outdir_m = os.path.join(exp_root, '{}.lsh64/matches'.format(exp_name) )

    outfiles = sorted(glob.glob(outdir_m + '/out.*'))
    if (len(outfiles)) == 0: 
        print('DIRECTORY ERROR', outdir_m)

        return 0
        
    out_df_list = []
    for name in outfiles:
        out_df_list.append( get_matches_df(name) )

        
    # append all matches df to a single df
    matches_df = pd.concat(out_df_list, ignore_index=True)

    return matches_df


def discovered_fragments(matches_df):
    F_disc = pd.DataFrame(columns=['filename', 'start', 'end'])
    for i, row in matches_df.iterrows():
        F_disc = F_disc.append([{'filename': row['f1'],
                                 'start': row['f1_start'],
                                 'end': row['f1_end']}], ignore_index=True)
        F_disc = F_disc.append([{'filename': row['f2'],
                                 'start': row['f2_start'],
                                 'end': row['f2_end']}], ignore_index=True)

    return F_disc


def plot_match_stats(matches_df):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 4))
    axes[0].hist(matches_df.score)
    axes[0].set_title('DTW scores')
    axes[1].hist(pd.concat([(matches_df.f2_end - matches_df.f2_start), (matches_df.f1_end - matches_df.f1_start)]),
                 bins=range(50))
    axes[1].set_title('fragment lengths distribution')
    axes[2].hist(matches_df.rho)
    axes[2].set_title('rho scores')


""" POST DISC DISPLAYING """

def new_node_dict(new_line):
    match = {'filename': str(new_line[0]),
             'start': int(new_line[1]),
             'end': int(new_line[2]),
             'score': float(new_line[3]),
             'unknown': float(new_line[4]),
             'file_id': int(new_line[5])
            }

    return match


def get_nodes_list(results_path):

    nodes_list = []

    nodes_file = open(results_path + '/master_graph.nodes', 'r')

    for i, line in enumerate(nodes_file):

        new_line = line.strip('\n').split('\t')

        nodes_list.append(new_node_dict(new_line))

    nodes_file.close()

    print('Read {} nodes/tokens'.format(len(nodes_list)))

    return nodes_list


def get_nodes_df(results_path):

    if results_path[-7:] != 'results':     # if exp_name is given
        results_path = join(zr_root, 'exp', results_path + '.lsh64', 'results')

    nodes_list = get_nodes_list(results_path)

    nodes_df = pd.DataFrame.from_records(nodes_list,
                                         columns=['filename', 'start', 'end', 'score', 'unknown', 'file_id']
                                         )

    nodes_df.index += 1

    return nodes_df


def get_clusters_list(results_path):

    if results_path[-7:] != 'results':     # if exp_name is given
        results_path = join(zr_root, 'exp', results_path + '.lsh64', 'results')

    clusters_list = []

    cluster_file = open(results_path + '/master_graph.dedups', 'r')

    for i, line in enumerate(cluster_file):
    #    print(line)
        new_line = line.strip('\n').split(' ')

        clusters_list.append([int(i) for i in new_line])

    cluster_file.close()

    print('Read {} clusters'.format(len(clusters_list)))

    return clusters_list


