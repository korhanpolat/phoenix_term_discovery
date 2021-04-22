
# coding: utf-8

# In[103]:
import cv2
import numpy as np
import os
from utils.helper_fncs import load_obj
from os.path import join
from scipy.signal import medfilt

# In[104]:
features_root = join('/home/korhan/Desktop/tez/dataset/features')
phase = 'train'


df = load_obj('train_labels')

signer_id = 'Signer06'

df = df.loc[df.signer != signer_id]
df = df.reset_index(drop=True)

folders_df = df.folder.value_counts()

w = 210
h = 260


# In[109]:


def get_hand(hand,frame):
    y = int(hand[1])
    x = int(hand[0])
    if x < 46:
        x = 46
    if x > 210 - 46:
         x = 210-46
    if y < 66:
        y = 66
    if y > 260-66:
        y = 260 - 66
    f = frame[y-66:y+66,x-46:x+46,:]
    f = cv2.resize(f,(92,132))
    return f

def detect_and_crop(img_path, xy_r, xy_l):

    img1 = cv2.imread(img_path)

    right_hand = get_hand(xy_r, img1.copy())
    left_hand = get_hand(xy_l, img1.copy())
    left_hand = np.fliplr(left_hand)
    
    return right_hand,left_hand
            
    
def get_img_list(source_path, folder_name):
    img_list = []
    path = join(source_path, folder_name)
    for d in sorted(os.listdir(path)):
        if d.endswith('.png'):
            img_list.append(join(path, d))
    return img_list


def process_keypoints(pose, hand):
    
    x1 = medfilt(pose[:,0])
    y1 = medfilt(pose[:,1])

    x2 = medfilt(hand[:,0])
    y2 = medfilt(hand[:,1])

#    weight_denom = hand[:,2] + pose[:,2]
#
#    x = (np.dot(x1,pose[:,2]) + np.dot(x2,hand[:,2])) / weight_denom
#    y = (np.dot(y1,pose[:,2]) + np.dot(y2,hand[:,2])) / weight_denom
    
    w1, w2 = (.3, .7)
    
    x = (x1 * w1 + x2 * w2)
    y = (y1 * w1 + y2 * w2) 
    
    xy = np.concatenate(( x[...,np.newaxis], y[...,np.newaxis] ), axis=1)
    
    return xy

    

def create_hands(source_path, target, phase, folders_df):


    for i in range(len(folders_df)):
        
        print(i)

        folder_name = folders_df.keys()[i]
        n = folders_df[i]
        
        img_list = get_img_list(source_path, folder_name)
        
        assert n == len(img_list)
        
        os.mkdir(join(target,'right',phase, folder_name))
        os.mkdir(join(target,'left',phase, folder_name))
        
        pose_kp = np.load(join('/home/korhan/Desktop/tez/dataset/features/op_body', phase, folder_name + '.npy'))
        hand_kp_r = np.load(join('/home/korhan/Desktop/tez/dataset/features/op_hand_kp', 'right', phase, folder_name + '.npy'))
        hand_kp_l = np.load(join('/home/korhan/Desktop/tez/dataset/features/op_hand_kp', 'left', phase, folder_name + '.npy'))

        xy_r = process_keypoints(pose_kp[:,4,:], hand_kp_r[:,0,:])
        xy_l = process_keypoints(pose_kp[:,7,:], hand_kp_l[:,0,:])

        
        for j,img_path in enumerate(img_list):
            
            hands = detect_and_crop(img_path, xy_r[j], xy_l[j])
            
            if hands[0] is not False:
                cv2.imwrite(join(target, 'right', phase, folder_name, os.path.split(img_path)[-1]), hands[0])
            if hands[1] is not False:
                cv2.imwrite(join(target, 'left', phase, folder_name, os.path.split(img_path)[-1]), hands[1])


# In[110]:
                
                
phase = 'train'
source_path = join('/home/korhan/Desktop/tez/dataset/features/fullFrame-210x260px',phase)
target = '/home/korhan/Desktop/tez/dataset/features/hand_crop'

#try:
#    shutil.rmtree(join(target))
#    os.mkdir('/home/pilab/Desktop/train')
#except:
#    os.mkdir('/home/pilab/Desktop/train')
    
create_hands(source_path, target, phase, folders_df)
























