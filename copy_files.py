#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 15:22:06 2018

@author: korhan
"""

import os
import shutil
from utils.db_utils import  get_labels_for_signer


def copy_files(source_path, dest_path, folders=None):

    os.makedirs(dest_path, exist_ok=True)
    
    if folders is None: folders = os.listdir(source_path)   
    
    for i in range(len(folders)):
    
        print(str(i)+'/'+str(len(folders)))
        src = os.path.join(source_path,folders[i],'1')
        dst = os.path.join(dest_path,folders[i])
        
        shutil.copytree(src, dst)


signer_id = 'Signer03' # 'Signer04' ten sonrasi memory hatasi
df = get_labels_for_signer(signer_id)

phase = 'train'
source_path = os.path.join(
        '/media/korhan/ext_hdd/tez_datasets/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px',
        phase)

dest_path = os.path.join('/home/korhan/Desktop/tez/dataset/features/fullFrame-210x260px',phase)


copy_files(source_path, dest_path, folders=list(df.folder.unique()))
