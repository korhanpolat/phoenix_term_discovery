#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 13:23:08 2018

@author: korhan
"""

import tf_pose
import cv2
import os
import json
import pickle
import shutil

def get_features(e,img_path):
    
    img = cv2.imread(img_path)
    humans = e.inference(img,upsample_size=4.0)
    human= humans[0]
    body_parts = human.body_parts    
    
    return body_parts

def save_json(full_path,data):
    with open(full_path+'.json', 'w') as fp:
        json.dump(data, fp, sort_keys=True, indent=4)
    
def save_pkl(full_path,data):
    with open(full_path+'.pkl', 'wb') as fp:
        pickle.dump(data, fp)


e = tf_pose.get_estimator()

phase = 'train'
source_path = os.path.join('/home/korhan/Desktop/tez/dataset/features/fullFrame-210x260px',phase)
dest_path = os.path.join('/home/korhan/Desktop/tez/dataset/features/openpose',phase)

try:
    shutil.rmtree(dest_path)
    os.mkdir(dest_path) 
except:
    os.mkdir(dest_path)

folders = os.listdir(source_path)


for i in range(len(folders)):

    print(str(i)+'/'+str(len(folders)))
    src = os.path.join(source_path,folders[i])

    img_files = sorted(os.listdir(src))
    
    try:
        shutil.rmtree(os.path.join(dest_path,folders[i]))
        os.mkdir(os.path.join(dest_path,folders[i])) 
    except:
        os.mkdir(os.path.join(dest_path,folders[i]))    
    
    for j in range(len(img_files)):
        
        img_path = os.path.join(src,img_files[j])
        feats = get_features(e,img_path)

        
        dst = os.path.join(dest_path,folders[i],img_files[j][:-4])
        save_pkl(dst,feats)        
        






















