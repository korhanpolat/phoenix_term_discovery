
from os import listdir
import numpy as np
from os.path import join, isfile
from os import chdir, makedirs, remove
import pandas as pd
from utils.db_utils import (img_paths_for_folder, fragment_tokenizer, gold_fragments_df_for_signer, get_labels_for_signer,
                            interpolate_garbage_labels, nodes_with_types, nodes_with_dom_labels, nodes_with_info, info_frames_for_segment)
from utils.ZR_utils import get_matches_df, get_nodes_df, get_clusters_list, plot_match_stats
import matplotlib.pyplot as plt
from shutil import rmtree, copyfile
import stringdist
import itertools
from collections import OrderedDict
from utils.helper_fncs import load_obj
from PIL import Image, ImageFont, ImageDraw
import time

zr_root = '/home/korhan/Desktop/zerospeech2017/track2/src/ZRTools'
lsh_path = '/home/korhan/Dropbox/tez/features/to_zrtools/lsh/tmp'


" EVALUATION "

"""
def matching_NED(df,matches_df, group=3, plot_hist=False):
    " percentage of phonemes shared by the two strings \
        normalization is done wrt frame counts "

    neds = np.zeros(len(matches_df))

    for i, row in matches_df.iterrows():
        filename, start, end = (row['f1'], row['f1_start'], row['f1_end'])
        labels1 = (df.label[df.folder == filename][start:end + 1] // group).values
        # in order not to count garbage classes from different sequences as the same label
        labels1[ labels1 == 3693//group] += 1

        filename, start, end = (row['f2'], row['f2_start'], row['f2_end'])
        labels2 = (df.label[df.folder == filename][start:end + 1] // group).values

        neds[i] = stringdist.levenshtein_norm(labels1, labels2)

    if plot_hist:
        plt.hist(neds)
        plt.title('Normalized Edit Distance Histogram')
        plt.show()

    return sum(neds) / len(neds)
"""

def strdist(source, target):

    s_range = range(len(source) + 1)
    t_range = range(len(target) + 1)
    matrix = [[(i if j == 0 else j) for j in t_range] for i in s_range]

    # Iterate through rest of matrix, filling it in with Levenshtein
    # distances for the remaining prefix combinations
    for i in s_range[1:]:
        for j in t_range[1:]:
            # Applies the recursive logic outlined above using the values
            # stored in the matrix so far. The options for the last pair of
            # characters are deletion, insertion, and substitution, which
            # amount to dropping the source character, the target character,
            # or both and then calculating the distance for the resulting
            # prefix combo. If the characters at this point are the same, the
            # situation can be thought of as a free substitution
            del_dist = matrix[i - 1][j] + 1
            ins_dist = matrix[i][j - 1] + 1
            sub_trans_cost = 0 if source[i - 1] == target[j - 1] else 1
            sub_dist = matrix[i - 1][j - 1] + sub_trans_cost

            # Choose option that produces smallest distance
            matrix[i][j] = min(del_dist, ins_dist, sub_dist)


    # At this point, the matrix is full, and the biggest prefixes are just the
    # strings themselves, so this is the desired distance
    distance = matrix[len(source)][len(target)]
    return float(distance) / max(len(source), len(target))


def matching_NED(gold_fragments, matches_df, plot_hist=False):
    " percentage of phonemes shared by the two strings \
    normalization is done wrt number of different labels instead of frame counts "
    if len(matches_df) == 0: return 1.0

    neds = np.zeros(len(matches_df))

    for i, row in matches_df.iterrows():

        filename, start, end = (row['f1'], row['f1_start'], row['f1_end'])
        labels1 = fragment_tokenizer(gold_fragments, filename, start, end)

        filename, start, end = (row['f2'], row['f2_start'], row['f2_end'])
        labels2 = fragment_tokenizer(gold_fragments, filename, start, end)

        try:
            neds[i] = stringdist.levenshtein_norm(labels1, labels2)
        except:
            neds[i] = strdist(labels1, labels2)

    if plot_hist:
        plt.hist(neds)
        plt.title('Normalized Edit Distance Histogram')
        plt.show()

    return sum(neds) / len(neds)


def clus_NED(gold_fragments, nodes_df, clusters_list ):
    " frameler uzerinden degil, transcription uzerinden hesaplaniyor "

    if len(nodes_df) == 0: return 1.0

    P_clus = []
    for clus in clusters_list:
        for pair in itertools.combinations(clus, 2):
            P_clus.append(list(pair))

    neds = np.zeros(len(P_clus))

    for i, pair in enumerate(P_clus):

        labels = []

        for p in pair:
            labels.append(nodes_df.types[p])

        # in order not to count garbage classes from different sequences as the same label
        if (len(labels[0]) == 0) | (len(labels[1]) == 0):
            neds[i] = 1.
        else:
            try:
                neds[i] = stringdist.levenshtein_norm(labels[0], labels[1])
            except:
                neds[i] = strdist(labels[0], labels[1])

#        neds[i] = stringdist.levenshtein_norm(labels[0], labels[1])

    return sum(neds) / len(neds)


def cluster_purity(df, exp_name=None, nodes_df=None, clusters_list=None,group=3, plot_th=-1, interp=False):
    """
    :type plot_th: float
    """

    if exp_name is not None: results_path = join(zr_root, 'exp', exp_name + '.lsh64', 'results')

    if clusters_list is None: clusters_list = get_clusters_list(results_path)
    if len(clusters_list) == 0: return 0.0, 0.0

    if nodes_df is None: nodes_df = get_nodes_df(results_path)


    G = np.zeros((3694//group+1, len(clusters_list)))

    for c, cluster in enumerate(clusters_list):
#        print(cluster)
        for i in range(len(cluster)):
            token_id = cluster[i]
            token = nodes_df.loc[token_id]
            label_counts = df.label[ df.folder == token.filename ][token.start:token.end].value_counts()
            for count in label_counts.iteritems():
                G[count[0]//group,c] += count[1]

    if not interp:
        G_nongarbage = G[:-1] # exclude garbage labels
        garbage_ratio = sum(G[-1]) / G.sum()

    else:
        G_nongarbage = G
        garbage_ratio = 0

    purity = sum(G_nongarbage.max(axis=0)) / G.sum()
    inv_purity = sum(G_nongarbage.max(axis=1)) / G.sum()

    #print('cluster purity: {:.3}, \t inverse purity: {:.3}'.format(purity, inv_purity))
    #print('Garbage frame ratio: {:.3}'.format(sum(G[-1]) / G.sum()))

    if plot_th > 0:
        nonzero_labels = np.array(np.where(~np.all(G < plot_th, axis=1))).squeeze().tolist()
        nonzero_labels = [df.label_name[df.label == label * group][0:1].item()[:-1] for label in nonzero_labels]

        G = G[~np.all(G < plot_th, axis=1)]  # delete zero rows
        G_norm = G / G.sum(axis=0)  # normalize clusters

        fig, ax = plt.subplots(1, 1, figsize=(int(G.shape[1] / 3), int(len(nonzero_labels) / 3)))
        im = ax.imshow(G_norm, cmap='Blues', vmin=0, vmax=1)
        ax.set_yticks(np.arange(len(nonzero_labels)))
        ax.set_yticklabels(nonzero_labels, fontsize=8, rotation=2)
        ax.set_xticks(np.arange(G.shape[1]))
        ax.set_xlabel('cluster IDs')
        ax.set_ylabel('word types')
        fig.colorbar(im, ax=ax, fraction=0.086, pad=0.04)

        for i in range(G.shape[0]):
            for j in range(G.shape[1]):
                if G[i, j] > 0:
                    text = ax.text(j, i, int(G[i, j]), ha="center", va="center", color="r")

    return purity, inv_purity, garbage_ratio


"""def number_of_discovered_frames(nodes_df):
        total_frames = 0
        n_found_frames = 0
        for filename in nodes_df.filename.unique():
            tmp = nodes_df.loc[nodes_df.filename == filename]
            total_frames += (tmp.end - tmp.start + 1).values[0]
            frames_found = np.zeros(tmp.end.max() + 1)
            for i, row in tmp.iterrows():
                frames_found[row['start']:row['end'] + 1] = 1
            n_found_frames += sum(frames_found)
    
        if(1 - n_found_frames / total_frames > .1): print('MORE THAN 10% OVERLAP')
        
        return n_found_frames
"""    

def number_of_discovered_frames(nodes_df, clusters_list, returnolap=False):
    nmax = nodes_df.end.max()+1

    discovered_counts = { key : np.zeros(nmax) for key in nodes_df.filename.unique() }
    for clus in clusters_list:
        tmp = nodes_df.loc[clus]
        for i, row in tmp.iterrows():
            discovered_counts[row['filename']][row['start']:row['end'] + 1] += 1
            
    n_found_frames = 0 # covered frames
    total_frames = 0 # number of total occurences
    for arr in discovered_counts.values():
        n_found_frames += sum(arr>0)
        total_frames += sum(arr)

    olapratio = 1 - n_found_frames / total_frames 

    if(olapratio > .1): print('{}% OVERLAP'.format(olapratio*100))

    if returnolap: return n_found_frames, olapratio
    else: return n_found_frames


def covered_frames(df, nodes_df):

    if len(nodes_df) == 0: return 0.0

    n_found_frames = number_of_discovered_frames(nodes_df)

    coverage = n_found_frames / len(df)

    return coverage


def coverage_no_singleton(gold_fragments, nodes_df, clusters_list):
    # coverage i singletonlari cikarip hesapla

    if len(nodes_df) == 0: 
        print('NO NODES FOUND')
        return 0.0

    non_single_set = gold_fragments.labelname.value_counts()[gold_fragments.labelname.value_counts() > 1].keys()
    non_single_gold_df = gold_fragments.loc[gold_fragments.labelname.apply(lambda x: x in non_single_set)]
    n_discoverable_frames = sum(non_single_gold_df.end - non_single_gold_df.start )

    n_found_frames = number_of_discovered_frames(nodes_df, clusters_list) * 1.

    coverage = n_found_frames / n_discoverable_frames
    
    return coverage



def precision_recall_F(n_common, n_disc, n_gold):
    if n_disc > 0:
        prec = n_common / float(n_disc)
    else:
        prec = 0

    if n_gold > 0:
        rec = n_common / float(n_gold)
    else:
        rec = 0

    if (rec > 0) and (prec > 0):
        F_score = 2 / (1 / prec + 1 / rec)
    else:
        F_score = 0.

    return (prec, rec, F_score)


def flat_pair(P):
    flat = set()
    for pair in P:
        flat |= {pair[0]}
        flat |= {pair[1]}
    return list(flat)


def pair_overlap(nodes_df, pair):
    " check whether a pair of nodes overlap "
    if nodes_df.filename[pair[0]] == nodes_df.filename[pair[1]]:
        non_overlap = (nodes_df.end[pair[0]] < nodes_df.start[pair[1]]) | (nodes_df.end[pair[1]] < nodes_df.start[pair[0]])
        if non_overlap: return False
        else: return True
    else: return False


def get_all_substrings(input_string):
    length = len(input_string)
    try:
        return [input_string[i:j + 1] for i in xrange(length) for j in xrange(i,length)]
    except:
        return [input_string[i:j + 1] for i in list(range(length)) for j in list(range(i, length))]


def matching_quality(nodes_df, clusters_list, gold_fragments):
    # substrings for found matches
    subs_for_node = dict()
    for i, row in nodes_df.iterrows():
        subs_for_node[str(i)] =set(get_all_substrings(row.types))

    # find P_clus (P_disc)
    P_clus = []
    for clus in clusters_list:
        for pair in itertools.combinations(clus, 2):
            P_clus.append(list(pair))

    # find P_disc*
    P_disc = set()
    for pair in P_clus:

        labels = [subs_for_node[str(pair[i])] for i in range(2)]
        for label1 in labels[0]:
            for label2 in labels[1]:
                filenames = sorted(nodes_df.filename[pair].values)
                P_disc |= {((filenames[0], label1), (filenames[1], label2))}

    # find P_all: set of all possible non overlapping matching fragment pairs
    filenames = sorted(gold_fragments.filename.unique())
    subs_for_folder = []
    P_all = set()

    for i in range(len(filenames)):
        tmp = gold_fragments.loc[gold_fragments.filename == filenames[i]].sort_values(by='start')
        # within file matches
        repeating_labels = tmp.label.value_counts()[tmp.label.value_counts() > 1]
        for row in repeating_labels.iteritems():
            P_all |= {(filenames[i], filenames[i], row[0])}

        subs_for_folder.append(set(get_all_substrings(tuple(tmp.label.values))))

    # for different files
    for i in range(len(filenames)):
        for j in range(i + 1, len(filenames)):
            matching_labels = subs_for_folder[i] & subs_for_folder[j]
            for label in matching_labels:
                P_all |= {((filenames[i], label), (filenames[j], label))}

    # calculate scores
    n_common = len(flat_pair(P_all & P_disc))
    n_disc = len(flat_pair(P_disc))
    n_gold = len(flat_pair(P_all))

    return precision_recall_F(n_common, n_disc, n_gold)


def clustering_quality(nodes_df, gold_fragments, clusters_list):
    if len(nodes_df) == 0: return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    P_clus = []
    for clus in clusters_list:
        for pair in itertools.combinations(clus, 2):
            P_clus.append(list(pair))

    # find the types that are repeated more than once, it also counts types that consist of 2 or more words
    gold_clus_types = nodes_df.types.value_counts().keys()[nodes_df.types.value_counts() > 1].values

    P_goldclus = []
    for typ in gold_clus_types:
        if len(typ) == 0: continue  # eliminate empty types from gold cluster
        clus = nodes_df.index[nodes_df.types == typ].values
        for pair in itertools.combinations(clus, 2):
            if not pair_overlap(nodes_df, pair):
                P_goldclus.append(list(pair))

    # since gold_clus_types are made up of discovered types, intersection of Pair sets will be intersection of types
    intersection = set(tuple(i) for i in P_clus) & set(tuple(i) for i in P_goldclus)

    grouping_q = precision_recall_F(len(flat_pair(intersection)), len(flat_pair(P_clus)), len(flat_pair(P_goldclus)))

    " type quality"
    gold_types = gold_fragments.label.value_counts().index[gold_fragments.label.value_counts() > 1]

    # old method
#    types = nodes_df.types.unique()
#    intersection = set(tuple(i) for i in types) & set(tuple([i]) for i in gold_types)

    # new method, find most dominant (more than 50% ) types for each cluster and count them as discovered type
    types = set()
    for clus in clusters_list:
        clus_nodes = nodes_df.loc[clus]
        type_id = clus_nodes.types.value_counts().index[ clus_nodes.types.value_counts(normalize=True) > .5 ]
        types |= set(type_id)

    intersection = types & set(tuple([i]) for i in gold_types)

    type_q = precision_recall_F(len(intersection), len(types), len(gold_types))

    #    print(((grouping_precision, grouping_recall, grouping_F_score),(type_precision, type_recall, type_F_score)))
    return (grouping_q, type_q)


def average_purity(clusters_list, nodes_df):
    # assign each cluster to dominant single label

    correct_mapped_cnt = 0
    for c, cluster in enumerate(clusters_list):

        lbl_count = np.zeros(len(cluster), dtype=np.int64)
        for i, node_id in enumerate(cluster):
            lbl_count[i] = nodes_df.labels_dom[node_id]

        lbl_count = lbl_count[lbl_count >= 0] # remove garbage classes

        if len(lbl_count) > 0:
            cluster_label = np.bincount(lbl_count).argmax()
            correct_mapped_cnt += sum(lbl_count == cluster_label)

    average_purity = float(correct_mapped_cnt) / len(nodes_df)

    return average_purity


def cluster_purity_types(clusters_list, nodes_df, plot_th=0):
    # assign each cluster to dominant sequence of gold types

    uniq_types = sorted(nodes_df.types.unique())
    G = np.zeros((len(uniq_types), len(clusters_list)))

    for c, cluster in enumerate(clusters_list):
        for i in range(len(cluster)):
            token_id = cluster[i]
            types = nodes_df.types[token_id]
            type_i = uniq_types.index(types)

            G[type_i, c] += 1

    if uniq_types[0] == ():
        notype = True
    else:
        notype = False

    if notype:
        G_nongarbage = G[1:]  # exclude garbage labels
    else:
        G_nongarbage = G
    purity = sum(G_nongarbage.max(axis=0)) / G.sum()
    inv_purity = sum(G_nongarbage.max(axis=1)) / G.sum()

    if plot_th > 0:

        labels_dict = load_obj('labels_dict_grouped', '/home/korhan/Dropbox/tez/files/')

        uniq_labelnames = [' ; '.join([labels_dict[str(t)] for t in types]) for types in uniq_types]
        if notype: uniq_labelnames[0] = '---'

        if plot_th == 1: G = G / G.sum(axis=0)  # normalize clusters

        reorder = np.argmax(G, axis=1)
        indexes = np.unique(reorder, return_index=True)[1]
        reorder = [reorder[index] for index in sorted(indexes)]
        reorder.extend(list(set(np.arange(G.shape[1])) - set(reorder)))

        Gd = G[:, reorder[:G.shape[1]]]

        fig, ax = plt.subplots(1, 1, figsize=(int(G.shape[1] / 3), int(len(uniq_types) / 3)))

        if plot_th == 1 or plot_th == 2:
            im = ax.imshow(Gd, cmap='Blues', vmin=0, vmax=1)
        else:
            im = ax.imshow(Gd, cmap=plt.cm.get_cmap('Blues', int(G.max()) + 1))

        ax.set_yticks(np.arange(len(uniq_types)))
        ax.set_yticklabels(uniq_labelnames, fontsize=8, rotation=2)
        ax.set_xticks(np.arange(G.shape[1]))
        ax.set_xticklabels(reorder)

        ax.set_xlabel('cluster IDs')
        ax.set_ylabel('word types')

        cbar = fig.colorbar(im, ax=ax,  fraction=0.086, pad=0.04)
        cbar.set_ticks(np.arange(0, int(G.max() + 1), 1))

#        if plot_th == 1:
#            for i in range(G.shape[0]):
#                for j in range(G.shape[1]):
#                    if G[i, j] > 0:
#                        text = ax.text(j, i, int(G[i, j]), ha="center", va="center", color="w")

    return purity, inv_purity



def unite_clusters(clusters_list):
    clusters_set = set()
    for row in clusters_list: clusters_set |= set(row)
        
    return list(clusters_set)
    

def purify_(tmp, discovered_counts):
    to_remove_idx = []
    for i, row in tmp.iterrows():
        f,s,e = row[['filename','start','end']]
        if sum(discovered_counts[f][s:e]) > 0.5*(e-s): # rm index
            to_remove_idx.append(i)
        else:
            discovered_counts[f][s:e] = 1
            

    return to_remove_idx


def purify_keep_longest(tmp):
    """remove overlapping segments, keeping the longest segment
    tmp: nodes_df
    """
    
    discovered_counts = { key : np.zeros(400) for key in tmp.filename.unique() }

    
    # find overlaps
    to_remove_dict = dict()
    for i, row in tmp.iterrows():
        f,s,e = row[['filename','start','end']]
        if len(np.nonzero(discovered_counts[f][s:e])[0])  > 0.5*(e-s): # rm index
            prev_vals = np.unique(discovered_counts[f][s:e])
            if len(prev_vals) == 1: slct = 0
            elif len(prev_vals) == 2: slct = 1
            elif len(prev_vals) > 2: 
                print('2 different overlaps')
                continue                
            i_first = int(prev_vals[slct])
            to_remove_dict[str(i_first)].append((i,f,s,e))
        else:
            discovered_counts[f][s:e] = i
            to_remove_dict[str(i)] = [(i,f,s,e)]

    to_rmv_list = []
    for k,tuples in to_remove_dict.items():
        if len(tuples) > 1: 
            to_rmv_list.append(tuples)

    # discard all except longest
    for idx,tpls in enumerate(to_rmv_list):
        tmp = []
        for (i,f,s,e) in tpls:
            tmp.append(e-s)

        to_keep = np.array(tmp).argmax()
        del tpls[to_keep]

    # save the indices only
    to_remove_idx = []
    for idx,tpls in enumerate(to_rmv_list):
        tmp = []
        for (i,f,s,e) in tpls: to_remove_idx.append(i)
            
    return to_remove_idx


def purify_clusters(clusters_list, nodes_df):

    all_centroids = unite_clusters(clusters_list)
    tmp = nodes_df.loc[all_centroids]

    to_remove_idx = purify_keep_longest(tmp)

    new_clusters = []
    for clus in clusters_list:
        tmp_clus = []
        for c in clus:
            if c not in to_remove_idx:
                tmp_clus.append(c)
        if len(tmp_clus) > 1: new_clusters.append(tmp_clus)

    all_centroids = unite_clusters(new_clusters)
    nodes_df = nodes_df.loc[all_centroids]

    return new_clusters, nodes_df



def evaluate(df, signer_id, matches_df, nodes_df, clusters_list, group, interp=False, boundary_th=0.5, 
                plot=False, plot2=0, verbose=False, fast_compute=False, purify=False ):
    """
    :type exp_name: str
    """
    scores = OrderedDict( [ ('n_match', 0), ('n_node', 0), ('n_clus', 0), ('mean_len', 0), ('mean_n_type', 0),
                            ('ned', 1.), ('coverage', 0.),
                            ('matching_P', 0.0), ('matching_R', 0.0), ('matching_F', 0.0),
                            ('purity', 0.), ('inv_purity', 0. ), ('garbage_ratio', 0.),
                            ('avg_purity', 0.), ('clus_purity', 0. ), ('clus_purity_inv', 0.),                            
                            ('grouping_P', 0.), ('grouping_R', 0.), ('grouping_F', 0.),
                            ('type_P', 0.), ('type_R', 0.), ('type_F', 0.) ] )
    """
    scores = {'n_match': 0, 'n_node': 0, 'n_clus': 0,
              'ned': 1., 'coverage': 0.,
              'matching_P': 0.0, 'matching_R': 0.0, 'matching_F': 0.0,
              'purity': 0., 'inv_purity': 0., 'garbage_ratio': 0.,
              'grouping_P': 0., 'grouping_R': 0., 'grouping_F': 0.,
              'type_P': 0., 'type_R': 0., 'type_F': 0. }
    """
    if verbose: 
        start = time.time()

    if df == None: df = get_labels_for_signer(signer_id)

    if interp: df = interpolate_garbage_labels(df)

    gold_fragments = gold_fragments_df_for_signer(signer_id, group=group, interp=interp)

    if len(matches_df) == 0: return scores
    elif matches_df is None: scores['n_match'] = 0
    else: scores['n_match'] = len(matches_df)

    " get nodes "
#    if len(nodes_df) == 0: return scores
#    else: scores['n_node'] = len(nodes_df)

    if verbose: 
        print('{:.2f} secs - loadings complete'.format(time.time() - start))
        start = time.time()


    if purify: clusters_list, nodes_df = purify_clusters(clusters_list, nodes_df)


    clusters_set = set()
    for row in clusters_list: clusters_set |= set(row)
    all_centroids = list(clusters_set)

    scores['n_node'] = len(all_centroids)

    # select only nodes that belong to a cluster (dedups?)
    nodes_df = nodes_df.loc[all_centroids]


    nodes_df = nodes_with_types(nodes_df, gold_fragments, thr=boundary_th)
    nodes_df = nodes_with_dom_labels(df, nodes_df, group=group)
    nodes_df = nodes_with_info(nodes_df,gold_fragments,interp)

    if verbose: 
        print('{:.2f} secs - nodes with info'.format(time.time() - start))
        start = time.time()


    # average fragment length
    scores['mean_len'] = (nodes_df.end - nodes_df.start + 1).mean()
    # average number of types in a fragment
    scores['mean_n_type'] = sum([len(nodes_df.types.values[i]) for i in range(len(nodes_df)) ]) / float(len(nodes_df))


    " clustering "


    if len(clusters_list) == 0:
        print('no clusters found')
        return scores
    else: scores['n_clus'] = len(clusters_list)

    if verbose: 
        print('{:.2f} secs - stats'.format(time.time() - start))
        start = time.time()

    if fast_compute:
        scores['matching_P'], scores['matching_R'], scores['matching_F'] = 0,0,0
    else:    
        scores['matching_P'], scores['matching_R'], scores['matching_F'] = matching_quality(nodes_df, clusters_list, gold_fragments)

    if verbose: 
        print('{:.2f} secs - matching_quality'.format(time.time() - start))
        start = time.time()


    scores['ned'] = clus_NED(gold_fragments, nodes_df, clusters_list)
#    scores['coverage'] = covered_frames(df, nodes_df)
    if verbose: 
        print('{:.2f} secs - NED'.format(time.time() - start))
        start = time.time()


    scores['coverage'] = coverage_no_singleton(gold_fragments, nodes_df, clusters_list)

    if verbose: 
        print('{:.2f} secs - coverage'.format(time.time() - start))
        start = time.time()


    scores['avg_purity'] = average_purity(clusters_list, nodes_df)


    if verbose: 
        print('{:.2f} secs - average_purity'.format(time.time() - start))
        start = time.time()


    scores['purity'], scores['inv_purity'], scores['garbage_ratio'] = cluster_purity(df, nodes_df=nodes_df,
                                                        clusters_list=clusters_list , group=group, plot_th=plot*0.1, interp=interp)

    if verbose: 
        print('{:.2f} secs - cluster_purity'.format(time.time() - start))
        start = time.time()


    scores['clus_purity'], scores['clus_purity_inv'] = cluster_purity_types(clusters_list, nodes_df, plot_th=plot2)

    if verbose: 
        print('{:.2f} secs - cluster_purity_types'.format(time.time() - start))
        start = time.time()


    grouping_quality, type_quality = clustering_quality(nodes_df, gold_fragments, clusters_list)

    if verbose: 
        print('{:.2f} secs - clustering_quality'.format(time.time() - start))
        start = time.time()


    scores['grouping_P'], scores['grouping_R'], scores['grouping_F'] = grouping_quality
    scores['type_P'], scores['type_R'], scores['type_F'] = type_quality


    return scores


def evaluate_ZR(df, exp_name, signer_id, group, interp=False, boundary_th=0.5, plot=False, exp_root=zr_root+'/exp' ):
    """

    :type exp_name: str
    """

    " get matches "
    match_file_path = join(exp_root, exp_name+'.lsh64', 'matches', 'out.1')
    matches_df = get_matches_df(match_file_path)

    if plot: plot_match_stats(matches_df)

    " get nodes "
    nodes_df = get_nodes_df(exp_name)

    " get clusters "
    clusters_list = get_clusters_list(exp_name)

    scores = evaluate(df, signer_id, matches_df, nodes_df, clusters_list,
                      group=group, interp=interp, boundary_th=boundary_th, plot=plot)

    return scores

""" SAVING THE FOUND SEGMENTS """

def save_frames_for_segment(df, path, foldername, start, end, img_root = '/home/korhan/Desktop/tez/dataset/features/fullFrame-210x260px/train'):

    makedirs(path)

    label_names = list(df.label_name[df.folder == foldername][start: end + 1])  # assume frames start from 0
    img_paths = img_paths_for_folder(img_root, foldername, start, end + 1)

    for j, img_path in enumerate(img_paths):
        copyfile(img_path,
                 join(path, img_path.split('/')[-1][:-4] + '_' + label_names[j] + '.png'))


def save_token_frames(df, nodes_df, clusters_list):
    " save the image sequences corresponding to discovered nodes "

    img_root = '/home/korhan/Desktop/tez/dataset/features/fullFrame-210x260px/train'
    dest_root = '/home/korhan/Dropbox/tez/match_node_imgs'

    rmtree(join(dest_root), ignore_errors=True)

    # collect all tokens to a set
    dedups_set = set()
    for row in clusters_list:
        dedups_set = dedups_set | set(row)

    for i, row in nodes_df.iterrows():

        if i not in dedups_set: continue

        node_name = 'node_' + str(i).zfill(2)

        save_frames_for_segment(df, path=join(dest_root, node_name),
                                foldername=row['filename'],
                                start=row['start'],
                                end=row['end'],
                                img_root=img_root)


def save_token_frames_per_cluster(df, nodes_df, clusters_list):
    " save the image sequences corresponding to discovered nodes "

    img_root = '/media/korhan/ext_hdd/tez_datasets/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px/train'
    #img_root = '/home/korhan/Desktop/tez/dataset/features/fullFrame-210x260px/train'
    dest_root = '/home/korhan/Dropbox/tez/match_node_imgs'

    rmtree(join(dest_root), ignore_errors=True)

    for c, clus in enumerate(clusters_list):

        clus_name = 'clus_' + str(c).zfill(2)

        for node_idx in clus:

            row = nodes_df.loc[node_idx]
            node_name = 'node_' + str(node_idx).zfill(3)

            save_frames_for_segment(df, path=join(dest_root, clus_name, node_name),
                                    foldername=row['filename'],
                                    start=row['start'],
                                    end=row['end'],
                                    img_root=img_root)



def plot_curve(ax, results, ax_dict, label):
    # results: list of score dicts
    data = np.zeros((len(results), 2))
    annots = []
    for i,score in enumerate(results):
        data[i,0] = score[ax_dict['x']]
        data[i,1] = score[ax_dict['y']]
        annots.append(score['dtw'])

    ax.plot(data[:,0], data[:,1], label=label)
    ax.set_xlabel(ax_dict['x'])
    ax.set_ylabel(ax_dict['y'])
    for i,ann in enumerate(annots):
        if i%2 == 1: continue
        ax.annotate(str(ann), 
             xy=(results[i][ax_dict['x']], results[i][ax_dict['y']]),  
             xycoords='data')


def img_with_text(img_path, text, pos=(50,220), font=ImageFont.truetype('Pillow/Tests/fonts/FreeSans.ttf', 20)):

    img = Image.open(img_path)
    ImageDraw.Draw(img).text(pos,  text,  (255, 255, 55), font=font )

    return img


def save_token_frames_per_cluster_as_GIF(df, nodes_df, clusters_list, n_max=20,
    img_root = '/media/korhan/ext_hdd/tez_datasets/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px/train'):

    #img_root = '/home/korhan/Desktop/tez/dataset/features/fullFrame-210x260px/train'
    dest_root = '/home/korhan/Dropbox/tez/match_node_gifs/'

    font = ImageFont.truetype('Pillow/Tests/fonts/FreeSans.ttf', 20)

    rmtree(join(dest_root), ignore_errors=True)

    for c, clus in enumerate(clusters_list):
    
        clus_name = 'clus_' + str(c).zfill(2)
        makedirs(join(dest_root, clus_name))

        for i,nod in enumerate(clus):
            labels, img_paths = info_frames_for_segment(df, nodes_df.filename[nod], nodes_df.start[nod], nodes_df.end[nod])

            node_imgs = []

            for k, path in enumerate(img_paths):
                img = img_with_text(path, text=labels[k][:-1])
                node_imgs.append(img)

            for k_rest in range(k,n_max):
    #            blank_img = Image.new('RGB', img.size, (255, 255, 255))
                node_imgs.append(img)

            gif_name = '_'.join([nodes_df.filename[nod] , str(nodes_df.start[nod]) ,  str(nodes_df.end[nod])])
            node_imgs[0].save(join(dest_root,clus_name, gif_name + '.gif'), format='GIF', append_images=node_imgs[1:], save_all=True, duration=50, loop=0)



"""
            foldername = row['filename']
            start = row['start']
            end = row['end']
            node_name = 'node_' + str(node_idx).zfill(3)

            makedirs(join(dest_root, clus_name, node_name))

            label_names = list(df.label_name[df.folder == foldername][start: end + 1])  # assume frames start from 0
            img_paths = img_paths_for_folder(img_root, foldername, start, end + 1)

            for j, img_path in enumerate(img_paths):
                copyfile(img_path,
                         join(dest_root, clus_name, node_name,
                              img_path.split('/')[-1][:-4] + '_' + label_names[j] + '.png'))

"""