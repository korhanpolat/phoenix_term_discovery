from scipy.signal import medfilt
from utils.db_utils import get_labels_for_signer
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import itertools
from utils.ZR_utils import (get_matches_df, post_disc, plot_match_stats, change_post_disc_thr,
                            get_nodes_df, get_clusters_list, get_matches_all )
from utils.feature_utils import (derive_openpose, derive_openpose25_gaussian, derive_op25_joint_dist_, op_100_meancenter,
    list2dict,        get_feats,  random_project,         op_100, normalize_frames, apply_softmax, apply_PCA, apply_Gauss_norm, apply_mean_center)
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.nn import functional as F
import torch.nn
from torchvision import datasets, transforms
import glob
import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_distances
from utils.feature_eval import sdtw_jit
import glob


def apply_medfilt(feats, w=3):
    res = []
    for arr in feats: res.append(medfilt(arr, w))
    return res
        


def random_segment_idx(feats, L):
    rnd_file = np.random.randint(len(feats))
    while len(feats[rnd_file]) <= L+1:
        rnd_file = np.random.randint(len(feats))
    
    try:
        rnd_frameid = np.random.randint(len(feats[rnd_file]) - L+1)
    except:
        rnd_frameid = 0
        print(rnd_file,len(feats[rnd_file]))
    
    return rnd_file, rnd_frameid


def random_segment(feats, L):
    
    rnd_file, rnd_frameid = random_segment_idx(feats, L)
    
    rand_array = feats[rnd_file][rnd_frameid : rnd_frameid+L]
    
    return rand_array


def get_feats_for_Siamese(feats,feat_pairs):
    # align pairs and sample random segments for pairs

    # DTW align pairs
    feat_pairs_aligned = []
    rand_feats = [] # rand segment for each corr pair

    for pair in feat_pairs:
        pair = [pair[p].reshape(pair[p].shape[0],-1) for p in range(2)]
        dist_mat = cosine_distances(pair[0], pair[1])

        path, cost, matrix = sdtw_jit(dist_mat, w=max(dist_mat.shape), 
                                      start=[0, 0], backtr_loc=(dist_mat.shape[0]-1,dist_mat.shape[1]-1))

        path = np.array(path)
        tmp = np.zeros((2,len(path),pair[0].shape[1]))
        tmp_rand = np.zeros((2,len(path),*pair[0].shape[1]))
        
        for fi in range(2):
            tmp[fi] = pair[fi][path[:,fi]]
            tmp_rand[fi] = random_segment(feats, len(path))
        
        feat_pairs_aligned.append(tmp)
        rand_feats.append(tmp_rand)
        
        
    # concat all features to single array
    feats_cat = np.concatenate(feat_pairs_aligned,1).transpose((1,0,2))
    feats_cat_rand = np.concatenate(rand_feats,1).transpose((1,0,2))
    print(feats_cat.shape, feats_cat_rand.shape)
    dim = feats_cat.shape[2]
    print(dim)

    return feats_cat, feats_cat_rand


def cor_feat_pairs(feats_dict, matches_df):
    # read arrays for pairs
    feat_pairs = []
    for i, row in matches_df.iterrows():
        tmp = []
        for f in ['f1','f2']:
            arr = feats_dict[row[f]][row[f+'_start']:row[f+'_end']]
            tmp.append(arr)
        feat_pairs.append(tmp)
        
    return feat_pairs

def gold_feat_pairs(feats_dict, gold_pairs):
    feat_pairs = []
    for pair in gold_pairs:
        tmp = []
        for p in pair:
            f,s,e = p
            arr = feats_dict[f][s:e]
            tmp.append(arr)
        feat_pairs.append(tmp)

    return feat_pairs

def random_project(feats, d=64):
    proj = np.random.normal(0, 1, (feats[0].shape[1],d))
    feats = apply_Gauss_norm(feats)
    for i,arr in enumerate(feats):
        feats[i] = np.dot(arr, proj)
    return feats



