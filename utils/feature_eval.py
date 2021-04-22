from os.path import join
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)
import itertools
from utils.db_utils import gold_fragments_df_for_signer, get_labels_for_signer
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from utils.sdtw_funcs import sdtw_jit
from utils.feature_utils import kl_symmetric_pairwise
from sklearn.metrics import pairwise_distances 


# find true pairs
def get_true_pairs(gold_df, p_max, random):

    P_gold = []
    uniq_labels = gold_df.labelname.unique()
    if random: np.random.shuffle(uniq_labels)
    
    try:
        for name in uniq_labels:
            gold_set = gold_df[['filename','start','end']][gold_df.labelname == name]

            if random: gold_set = gold_set.sample(frac=1)

            for pair in itertools.combinations(gold_set.transpose(),2):

                P_gold.append([list(gold_set.loc[pair[0]].values),list(gold_set.loc[pair[1]].values)] )
                if len(P_gold) >= p_max: return P_gold
    except Exception as e: 
        print(e)
        pdb.set_trace()
                
    return P_gold


def get_false_pairs(gold_df, p_max, random):
    
    P_false = []
    if random: gold_df = gold_df.sample(frac=1)
    
    for random_pair in itertools.combinations(gold_df.transpose(),2):

        f0 = gold_df.loc[random_pair[0]]
        f1 = gold_df.loc[random_pair[1]]
        if f1.label != f0.label:
            P_false.append([[f0.filename,f0.start,f0.end],[f1.filename,f1.start,f1.end]])
            if len(P_false) >= p_max: return P_false

    return P_false


def pairwise_sdtw(feats0, feats1, loss_func='cosine'):

    if loss_func in ['cosine', 'euclidean', 'hamming']:
        dist_mat = pairwise_distances(feats0, feats1, loss_func)
    elif 'kl' in loss_func:
        dist_mat = kl_symmetric_pairwise(feats0, feats1)
    else: raise Exception('Not a valid loss function: {}'.format(loss_func))


    path, cost, matrix = sdtw_jit(dist_mat, w=max(dist_mat.shape), start=[0, 0], backtr_loc=(dist_mat.shape[0]-1,dist_mat.shape[1]-1))

    return cost / len(path)


def avg_cost_pair_set(feats, seq_names, pairs_set, loss_func='euclid'):
    assert len(feats) == len(seq_names)

    avg_cost = 0

    for pair in pairs_set:
        id0 = list(seq_names).index(pair[0][0])
        id1 = list(seq_names).index(pair[1][0])

        feats0 = feats[id0][pair[0][1]:pair[0][2]]
        feats1 = feats[id1][pair[1][1]:pair[1][2]]

        avg_cost += pairwise_sdtw(feats0, feats1, loss_func)

    avg_cost = avg_cost / len(pairs_set)
    
    return avg_cost



def feature_eval(feats, gold_df, seq_names, p_max, loss_func='euclid', random=True):
    
    P_gold = get_true_pairs(gold_df, p_max, random)
    p_max = len(P_gold)

    P_false = get_false_pairs(gold_df, p_max, random)
        
    cost_p = avg_cost_pair_set(feats, seq_names, P_gold, loss_func=loss_func)
    cost_n = avg_cost_pair_set(feats, seq_names, P_false, loss_func=loss_func)
    
    return cost_p / (cost_p + cost_n) 
    

def run_eval(feats, gold_df, seq_names, loss_func='euclid', n=30, p_max = 500):

    eval_scores = np.zeros((n,1))

    for i in range(n):
        eval_scores[i] = feature_eval(feats, gold_df, seq_names, p_max, loss_func=loss_func, random=True)

    return eval_scores.mean(), eval_scores.std()




# ABX TEST
def get_test_df(gold_df, clip_len):
    test_df = gold_df.loc[ gold_df.end - gold_df.start >= clip_len ].copy()
    test_df.reset_index(inplace=True, drop=True)        
    return test_df


def find_triplets(test_df, max_n=5000):
    ax_df = (test_df.groupby('labelname').filter(lambda x: len(x) > 1))
    gold_labels = sorted(ax_df.labelname.unique())
    np.random.shuffle( gold_labels )

    cnt = 0

    bax_pairs = []
    test_set = set()

    for label in gold_labels:
        for pair in itertools.combinations( ax_df.loc[ax_df.labelname == label].transpose(),2 ):

            ax = [tuple(ax_df.loc[pair[p]][['filename','start','end','labelname']].values) for p in [0,1]]

            b_row_id = np.random.choice( test_df.loc[test_df.labelname != label].index )
            bax_pairs.append( [tuple(test_df.loc[b_row_id][['filename','start','end','labelname']])] + ax )

            if len(bax_pairs) == max_n: break

        else:
            continue
        break

    for triplet in bax_pairs:
        test_set |= set(triplet)

    test_list = sorted(list(test_set))
    print(len(test_list))

    return bax_pairs, test_list


def get_feats_dict(seq_names, feats):
    feats_dict = dict()

    for i,arr in enumerate(feats):
        feats_dict[seq_names[i]] = arr

    return feats_dict


class ABX:

    def __init__(self, gold_df, min_len=8, max_n=5000):
        self.gold_df = gold_df
        self.clip_len = min_len
        self.test_df = get_test_df(gold_df, min_len)
        
        self.bax_pairs = []
        for i in range(10):
            bax_pairs, test_list = find_triplets(self.test_df, max_n=max_n/10)
            self.bax_pairs.append(bax_pairs)

   
    def calculate_score(self, feats_dict, loss = 'euclid'):
        results = np.zeros((len(self.bax_pairs)))
        
        for i,bax_pairs in enumerate(self.bax_pairs):
            score_sum = 0
            for triplet in bax_pairs:
                b, a, x = triplet

                dist_b = pairwise_sdtw(feats_dict[b[0]][b[1]:b[2]], feats_dict[x[0]][x[1]:x[2]], loss)
                dist_a = pairwise_sdtw(feats_dict[a[0]][a[1]:a[2]], feats_dict[x[0]][x[1]:x[2]], loss)
                if dist_a < dist_b: score_sum += 1
            
            results[i] = score_sum / len(bax_pairs)

        return results