import numpy as np
from os.path import join
import pandas as pd
import os
from utils.ZR_utils import run_disc_ZR
import string


def int2str_base62(x):

    digs = sorted(string.digits + string.ascii_letters)
    base = len(digs)
    assert base is 62
    
    digits = []
    while x:
        digits.append(digs[int(x % base)])
        x = int(x / base)

    digits.reverse()

    return ''.join(digits)


def str2int_base62(x):
    
    digs = sorted(string.digits + string.ascii_letters)
    base = len(digs)
    assert base is 62

    x = list(x)
    res, p = 0, 0

    while len(x) > 0:
        res += digs.index(x.pop())*base**p 
        p += 1
        
    return res



def concat_dict(dict_tobe_cat):
    names = []
    arrays = []
    lengths = []
    
    for key, arr in dict_tobe_cat.items():
        names.append(key)
        arrays.append(arr)
        lengths.append(len(arr))

    cat_arrays = np.concatenate(arrays,axis=0)

    return names, cat_arrays, lengths


def concat_whole_input(feats_dict, n_concat=5, seq_names=None, seq_names_ref=''):

    if seq_names is None: seq_names = list(feats_dict.keys())

    # number of sentences to concat
    n_seq = int(np.ceil(len(seq_names) / n_concat))

    feats_cat = dict()
    length_traceback = dict()

    for k in range(n_seq):
        dict_tobe_cat = {key: feats_dict[key] for key in seq_names[k::n_seq]}
        names, cat_arrays, lengths = concat_dict(dict_tobe_cat)
        
        newkey = 'files_' + '_'.join(
            [int2str_base62(seq_names_ref.index(name)) for name in names])

        feats_cat[newkey] = cat_arrays
        length_traceback[newkey] = lengths

    return feats_cat, length_traceback, seq_names



def retrieve_index(x, lengths):

    cum_sum = np.cumsum(lengths)
    idx = np.argmax((cum_sum-x)>0)
    true_timeidx = x - sum(lengths[:idx])

    return idx, true_timeidx


def discard_match(interval_info, lengths):
    se = 0
    se_list = []
    se_list.append((interval_info[se][1],lengths[interval_info[se][0]]))
    se = 1
    se_list.append((0, interval_info[se][1]))


    durations = [abs(s-e) for s,e in se_list]
    ratio = abs(durations[0] - durations[1]) / (durations[0] + durations[1])
    if ratio >= 0.5:
        idx = np.argmax(durations)
        return (interval_info[idx][0], *se_list[idx])
    else:
        return -1,0,0


def retrieve_interval_info(newkey, s, e, length_traceback):
    # given newkey and interval, recover original key and interval
    interval_info = [retrieve_index(x, length_traceback[newkey]) for x in [s,e]]

    if interval_info[0][0] != interval_info[1][0]:
        # discard interval if it overlaps two sentences
        keyid, s, e = discard_match(interval_info, length_traceback[newkey])
        if keyid == -1:
            return 'discard_match', 0, 0
        else:
            return newkey.split('_')[1:][keyid], s, e
    else: 
        key = newkey.split('_')[1:][interval_info[0][0]]
        s,e = [interval_info[se][1] for se in [0,1]]
        return key, s, e
    
    
def retrieve_matches(matches_df_cat, length_traceback, seq_names_ref):

    # prepare df, new columns
    f1cols = ['f1', 'f1_start', 'f1_end']
    f2cols = ['f2', 'f2_start', 'f2_end']
    matches_df_cat.rename(columns={x:x+'_cat' for x in f1cols + f2cols}, inplace=True)
    
    # retrieve original info for f1 intervals
    matches_df_cat[f1cols[0]], matches_df_cat[f1cols[1]], matches_df_cat[f1cols[2]]  = zip(
        *matches_df_cat[[x+'_cat' for x in f1cols]].apply(
            lambda x: retrieve_interval_info(*x, length_traceback), axis=1) )

    # discard if interval overlaps two files
    matches_df_cat = matches_df_cat[ ~matches_df_cat['f1'].isin(['discard_match']) ].copy()

    # retrieve original info for f2 intervals
    matches_df_cat[f2cols[0]], matches_df_cat[f2cols[1]], matches_df_cat[f2cols[2]]  = zip(
        *matches_df_cat[[x+'_cat' for x in f2cols]].apply(
            lambda x: retrieve_interval_info(*x, length_traceback), axis=1) )

    # discard if interval overlaps two files
    matches_df_cat = matches_df_cat[ ~matches_df_cat['f2'].isin(['discard_match']) ].copy()

    # convert file idx to names
    matches_df_cat['f1'] = matches_df_cat['f1'].apply(lambda x: seq_names_ref[str2int_base62(x)])
    matches_df_cat['f2'] = matches_df_cat['f2'].apply(lambda x: seq_names_ref[str2int_base62(x)])
    # clean dataframe
    matches_df_cat.drop([x+'_cat' for x in f1cols + f2cols], axis=1, inplace=True)
    matches_df_cat.reset_index(drop=True, inplace=True)    
    
    return matches_df_cat





def run_disc_ZR_feats_concat(feats_dict, params, seq_names=None, n_concat=5):
    # concat arrays, run discovery and seperate on matches df

    if 'n_concat' in params['disc'].keys(): n_concat = params['disc']['n_concat']
    # if 'uniq_id' in params.keys(): 
    #     uniq_id = params['uniq_id']
    # else:
    #     uniq_id = ''
    if params['dataset'] == 'phoenix':
        with open(join(params['CVroot'], 'all' + '.txt'), 'r') as f: 
            seq_names_ref = [x.strip('\n') for x in f.readlines()]


    feats_cat, length_traceback, seq_names = concat_whole_input(feats_dict, n_concat, seq_names, seq_names_ref)

    # params['disc']['B'] *= n_concat * 2

    matches_df_cat = run_disc_ZR(feats_cat, params)

    matches_df = retrieve_matches(matches_df_cat, length_traceback, seq_names_ref)
    
    return matches_df