#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import numpy as np
from numba import jit, njit, typeof
import matplotlib.pyplot as plt
from os.path import join
import pandas as pd

import os

# In[3]:


from utils.pipeline import run_matches_discovery, discovery_pipeline
from utils.pipeline_wrappers import gen_expname, run_until_coverage_th, try_run_exp, cv_experiment, monitor_callback


# In[4]:


from utils.helper_fncs import load_json
params = load_json(full_path='/home/korhan/Dropbox/config/knn.json')
params


# In[5]:


with open(join(params['CVroot'], params['CVset'] + '.txt'), 'r') as f: 
    seq_names = [x.strip('\n') for x in f.readlines()]
    
print(len(seq_names))


# In[6]:


# get feats dict
feats_dict = {}
for name in seq_names: 
    arr = np.load(join(params['feats_root'], 'deep_hand/c3/right/train/', name + '.npy'))
    feats_dict[name] = arr / arr.std(1)[:,None]

# feats_dict = fea_util.apply_PCA_dict(feats_dict, 0.99, whiten=True)
params['featype'] = 'c3right'

feats_dict[name].shape


# In[7]:


params['clustering']['olapthr_m'] = 0.2
params['clustering']['cost_thr'] = 0.2
params['covth'] = 0.1
params['basename'] = 'knn_deneme'
params['expname'] = gen_expname(params)
params['csvname'] = 'knn_exps'
params['csvname_factors'] = 'knn_search'
params['expname']


# In[8]:


# matches_df, seq_names = run_matches_discovery(feats_dict, params)


# # In[9]:


# matches_df, nodes_df, clusters_list, scores = discovery_pipeline(feats_dict, params)


# # In[39]:


# matches_df, nodes_df, clusters_list, scores, pars = run_until_coverage_th(
#     feats_dict, params, covth=0.1, covmargin=0.01)


# # In[7]:


# results = try_run_exp(feats_dict , params)


# # In[25]:


def save_csv(params, tmp, name):
    outfile = join(params['exp_root'], 'results', name + '_expresults.csv')
    if os.path.exists(outfile):
        pd.DataFrame([tmp]).to_csv(outfile, mode='a', header=False)
    else: 
        pd.DataFrame([tmp]).to_csv(outfile, mode='w', header=True)


# In[26]:


def run_factor_cv(feats_dict , ptype, key, vals, params, evals):
    tmp = []
    default = params[ptype][key]
    for val in vals:
        params[ptype][key] = val
        scores = cv_experiment(seq_names, feats_dict, params, nfold = 5)
        tmp.append(scores)
        save_csv(params, scores, name='factors2')


    evals.extend(tmp)
    params[ptype][key] = default

    return params


# In[21]:


params['disc']['a'] = 4
params['disc']['dim_fix'] = 8
params['disc']['metric'] = 'L2'
# params['disc']['lmin'] = 4
# params['disc']['use_gpu'] = False
params['disc']


# In[27]:





# In[22]:


params['basename0'] = 'knn_factors_set{}_{}'.format(params['CVset'], params['featype'])
scores = cv_experiment(seq_names, feats_dict, params, nfold = 5)
save_csv(params, scores, name='factors2')

# In[ ]:


evals = []
# params['basename0'] = 'knn_factors_set{}_{}'.format(params['CVset'], params['featype'])

# evals.append( cv_experiment(seq_names, feats_dict, params, nfold = 5) )
evals.append(scores)

params = run_factor_cv(feats_dict ,'disc', 'a',[3,5], params, evals)
params = run_factor_cv(feats_dict ,'disc', 'dim_fix',[4,6,10], params, evals)
params = run_factor_cv(feats_dict ,'disc', 'k',[50,150,200], params, evals)
params = run_factor_cv(feats_dict ,'disc', 'lmax',[20,30], params, evals)
params = run_factor_cv(feats_dict ,'disc', 'r',[0.01,0.05,0.1,0.2,0.5], params, evals)
params = run_factor_cv(feats_dict ,'disc', 's',[0.05,0.2,0.5,0.7], params, evals)
params = run_factor_cv(feats_dict ,'disc', 'top_delta',[0.005,0.01,0.05], params, evals)
params = run_factor_cv(feats_dict ,'disc', 'pca',['PCAW40'], params, evals)
params = run_factor_cv(feats_dict ,'disc', 'metric',['IP'], params, evals)


# In[ ]:




