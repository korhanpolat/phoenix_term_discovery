from utils.eval import evaluate


from joblib import Parallel, delayed
from itertools import combinations


from utils.sdtw_funcs import sdtw_jit, LCMA_jit, joints_loss, sdtw_np, LCMA_jit_new

from utils.ZR_utils import new_match_dict, change_post_disc_thr, post_disc2, get_nodes_df, get_clusters_list, run_disc_ZR, get_matches_all
from utils.ZR_cat import run_disc_ZR_feats_concat

import pandas as pd
import numpy as np
from utils.feature_utils import get_features_array_for_seq_name, get_paths_for_features, normalize_frames, kl_symmetric_pairwise
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances, cosine_similarity
from utils.eval import evaluate, save_token_frames_per_cluster, number_of_discovered_frames
from utils.helper_fncs import save_obj, load_obj
from os.path import join
import os
import traceback
from utils.discovery_utils import *

from numba import jit, prange



@jit(nopython=True)
def dist_to_edge_weight(distortion, thr):
    return max(0, (thr - distortion) / thr)


@jit
def gen_seeds(mat_shape, w=8, overlap=False):
    if not overlap: w = 2 * w

    seeds = []
    for k in range(int(np.floor((mat_shape[0] - 1) / (w + 1)))): seeds.append((k * (w + 1), 0))
    for k in range(1, int(np.floor((mat_shape[1] - 1) / (w + 1)))): seeds.append((0, k * (w + 1)))

    return seeds


@jit
def sdtw_min_paths_jit(dist_mat, w, seeds):
    # returns best paths for each seed
    paths = []
    for seed in seeds:
        path, cost, matrix = sdtw_jit(dist_mat, w=w, start=seed)
        paths.append(path)

    return paths



def compute_pairwise_distmat(feats0, feats1, loss_func):
    if 'euclid' in loss_func:
        dist_mat = euclidean_distances(feats0, feats1)
    elif 'cosine' in loss_func:
        dist_mat = cosine_distances(feats0, feats1)
    elif 'log_cosine' in loss_func:
        dist_mat = -np.log(1e-8 + cosine_similarity(feats0, feats1) )
    elif 'kl' in loss_func:
        dist_mat = kl_symmetric_pairwise(feats0, feats1)

    return dist_mat


def run_disc_given_distmat(dist_mat, fs, params):

    f0,f1 = fs
    
    matches_info = []
    
    seeds = gen_seeds(dist_mat.shape, params['w'], overlap=params['diag_olap'])

    paths = sdtw_min_paths_jit(dist_mat, params['w'], seeds)

    for k, path in enumerate(paths):

        path_distortion = dist_mat[[pair[0] for pair in path], [pair[1] for pair in path]]
        s, e, cost = LCMA_jit_new(path_distortion, params['L'], 
                              extend_r=params['extend_r'], 
                              end_cut=params['end_cut'])

        s0, e0 = path[s][0], path[e][0]
        s1, e1 = path[s][1], path[e][1]
        #            edge_w = dist_to_edge_weight(cost, thr)

        if abs((e1 - s1) - (e0 - s0)) / float(e - s) < params['diag_thr']:
            matches_info.append(new_match_dict((f0, f1),
                                               (s0, e0, s1, e1, cost)))

    return matches_info


def run_disc_single_pair( fs, feats_dict, params):

    f0,f1 = sorted(fs)
    
    dist_mat = compute_pairwise_distmat(feats_dict[f0], 
                                        feats_dict[f1], 
                                        params['loss_func'])

    matches_info = run_disc_given_distmat(dist_mat, (f0,f1), params)

    return matches_info


def run_disc_pairwise(feats_dict, params):
    seq_names = sorted(feats_dict.keys())

    matches = Parallel(n_jobs=params['njobs'])(
                    delayed(run_disc_single_pair)((f0,f1), 
                                                  feats_dict, params['disc']) 
                                        for f0,f1 in combinations(seq_names,2) )

    matches = [item for sublist in matches for item in sublist]

    return matches
