
import numpy as np
from os.path import join,dirname,abspath
from os import listdir
from sklearn.decomposition import PCA
import os
from numba import jit
from scipy.signal import medfilt2d


def get_seq_names(params):
    with open(join(params['CVroot'], params['CVset'] + '.txt'), 'r') as f: 
        seq_names = [x.strip('\n') for x in f.readlines()]

    return seq_names


def list2dict(seq_names, feats):
    feats_dict = dict()
    for i,name in enumerate(seq_names):
        feats_dict[name] = feats[i]

    return feats_dict

@jit
def kl_divergence(p, q):
    return sum(p * np.log2(p/q))

@jit
def kl_symmetric_pairwise(feats0, feats1):
    m,n = feats0.shape[0], feats1.shape[0]
    mat = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            mat[i,j] = (kl_divergence(feats0[i], feats1[j]) + kl_divergence(feats1[j], feats0[i])) / 2

    return mat


def apply_PCA(feats, exp_var, whiten=False, unitvar=True):

    X = np.concatenate(feats, axis=0)
    # unit variance each time vector
    if unitvar: X /= X.std(1)[:,None]

    pca = PCA(n_components=exp_var, whiten=whiten, random_state=42)
    pca.fit(X)

    Xt = pca.transform(X)

    res = []
    idx = 0
    for arr in feats:
        n = arr.shape[0]
        res.append(Xt[idx:idx + n, :])
        idx += n

    return res


def apply_PCA_dict(feats_dict, exp_var, whiten=False, unitvar=True):

    names = [] 
    feats = []
    for k,v in feats_dict.items():
        names.append(k)
        feats.append(v)

    res = apply_PCA(feats, exp_var, whiten, unitvar)

    for k,v in feats_dict.items():
        idx = names.index(k)
        feats_dict[k] = res[idx]

    return feats_dict


def get_feats(root_dir, seq_names, unitvar=True):

    feats = []
    for i, name in enumerate(sorted(seq_names)):

        X = np.load(os.path.join(root_dir,name + '.npy'))
        if unitvar: X /= X.std(1)[:,None]
        feats.append(X)

    return feats


def get_feats_dict(root_dir, seq_names, unitvar=True):

    feats = dict()
    for i, name in enumerate(sorted(seq_names)):
        
        X = np.load(os.path.join(root_dir,name + '.npy'))
        if unitvar: X /= X.std(1)[:,None]
        feats[name] = X

    return feats


def mean_center(array): # zero mean each feature dimension
    return (array - array.mean(axis=0))


def apply_mean_center(feats):
    X = np.concatenate(feats, axis=0)

    Xt = mean_center(X)

    res = []
    idx = 0
    for arr in feats:
        n = arr.shape[0]
        res.append(Xt[idx:idx + n, :])
        idx += n

    return res


def op_100_single(name, apply_medfilt=False, feats_root='/home/korhan/Desktop/tez/dataset/features/'):

    arr_pose = np.load(os.path.join(feats_root,'op_body',name + '.npy'))[:,:9,:2]
    shoulder_L = np.linalg.norm(arr_pose[:,5]-arr_pose[:,2],2,1).mean()
    neck = arr_pose[:,1,:2][:,None,:]
    wrist_L = arr_pose[:,7,:2][:,None,:]
    wrist_R = arr_pose[:,4,:2][:,None,:]

    arr_pose = np.delete(arr_pose, 1 , axis=1) # delete the neck coordinate, cuz uninformative
    
    arr = np.hstack(((arr_pose - neck) / shoulder_L,
                     (np.load(os.path.join(feats_root,'op_hand/right',name + '.npy'))[:,:,:2] - wrist_R) / shoulder_L,
                     (np.load(os.path.join(feats_root,'op_hand/left',name + '.npy'))[:,:,:2] - wrist_L) / shoulder_L
                    ))
    arr = arr.reshape(-1,100)
    if apply_medfilt: arr = medfilt2d(arr, kernel_size=(3,1))

    return arr


def op_100(seq_names, as_dict=False, apply_medfilt=False, feats_root='/home/korhan/Desktop/tez/dataset/features/'):

    
    if as_dict: 

        fdict = dict()

        for i,name in enumerate(seq_names):
            fdict[name] = op_100_single(name, apply_medfilt, feats_root)

        return fdict

    else: 
        
        feats = []
        
        for i,name in enumerate(seq_names):
            feats.append( op_100_single(name, apply_medfilt, feats_root) )

        return feats




def op_55_single(name, conf=False, apply_medfilt=False, feats_root=None):

    if conf: i=3
    else: i=2


    arr_pose = np.load(os.path.join(feats_root,'op_body',name + '.npy'))
    arr_pose = np.hstack((arr_pose[:, :9, :i], arr_pose[:, 15:19, :i]))

    shoulder_L = np.linalg.norm(arr_pose[:,5,:2]-arr_pose[:,2,:2],2,1).mean()
    neck = arr_pose[:,1,:2][:,None,:]
    
    arr = np.hstack(((arr_pose - neck) / shoulder_L,
                     (np.load(os.path.join(feats_root,'op_hand/right',name + '.npy'))[:,:,:i] - neck) / shoulder_L,
                     (np.load(os.path.join(feats_root,'op_hand/left',name + '.npy'))[:,:,:i] - neck) / shoulder_L
                    ))

    if apply_medfilt: arr = medfilt2d(arr, kernel_size=(3,1))

    return arr




def op_55_2(seq_names, as_dict=False, apply_medfilt=False, conf=False, 
            feats_root='/home/korhan/Desktop/tez/dataset/features/'):

    if as_dict: 

        fdict = dict()

        for i,name in enumerate(seq_names):
            fdict[name] = op_55_single(name, conf, apply_medfilt, feats_root)

        return fdict

    else: 
        
        feats = []
        
        for i,name in enumerate(seq_names):
            feats.append( op_55_single(name, conf, apply_medfilt, feats_root) )

        return feats


def op_100_meancenter(seq_names):
        
    feats = op_100(seq_names)

    feats = apply_mean_center(feats)

    return feats



def get_dh_feats(seq_names, base, hand, params):

    feats_dict = {}
    for name in seq_names: 
        if hand == 'both':
            arrs = [np.load(join(params['feats_root'], '{}/{}/train/'.format(
                                       base,h), name + '.npy')) for h in ['right','left']]
            for a,arr in enumerate(arrs):
                arrs[a] = arr / arr.std(1)[:,None]
            arr = np.concatenate(arrs, axis=1)
        else:
            arr = np.load(join(params['feats_root'], '{}/{}/train/'.format(base,hand), name + '.npy'))

        feats_dict[name] = arr

    return feats_dict


def get_dh_feats_wrap(seq_names, featype, params):
    
    if 'PCA40' in featype:
        base = 'deep_hand_PCA40'
    else:
        base = 'deep_hand'
    
    if 'c3' in featype:
        base = base + '/c3'
    elif 'l3' in featype:
        base = base + '/l3'

    if 'right' in featype: 
        hand = 'right'
    if 'left' in featype:
        hand = 'left'
    if 'both' in featype:
        hand = 'both'    

    return get_dh_feats(seq_names, base, hand, params)


def load_feats(seq_names, featype, params):
    
    if 'op' in featype:
        feats_dict = op_100(seq_names, as_dict=True, apply_medfilt=True)
    
    elif ('dh' in featype) or ('c3' in featype):
        feats_dict = get_dh_feats_wrap(seq_names, featype, params)

    if ('PCA' in featype) and (('dh' not in featype) and ('c3' not in featype)):
        if 'exp_var' in params.keys(): exp_var = params['exp_var']
        else: exp_var = 0.99

        feats_dict = apply_PCA(feats_dict, exp_var, whiten='W' in featype, unitvar='V' in featype)

    return feats_dict


def softmax_2d(x):
    # x is [t,d] array, softmax is applied for each t
    return np.exp(x) / np.exp(x).sum(axis=1)[:,None]


def apply_softmax(feats):

    for i,arr in enumerate(feats):
        feats[i] = softmax_2d(arr)

    return feats


def apply_L2norm(feats):

    for i,arr in enumerate(feats):
        feats[i] = arr / np.linalg.norm(arr,1)

    return feats


def normalize_frames_gaus(array): # zero mean unit variance each feature dimension
    return (array - array.mean(axis=0)) / (array.std(axis=0 ) + 0.001)


def normalize_frames(array):

    return normalize_frames_gaus(array)


def apply_Gauss_norm(feats):
    X = np.concatenate(feats, axis=0)

    Xt = normalize_frames_gaus(X)

    res = []
    idx = 0
    for arr in feats:
        n = arr.shape[0]
        res.append(Xt[idx:idx + n, :])
        idx += n

    return res



def apply_gausnorm_dict(feats_dict):

    names = [] 
    feats = []
    for k,v in feats_dict.items():
        names.append(k)
        feats.append(v)

    res = apply_Gauss_norm(feats)

    for k,v in feats_dict.items():
        idx = names.index(k)
        feats_dict[k] = res[idx]

    return feats_dict


def random_project(feats, d=64):
    proj = np.random.normal(0, 1, (feats[0].shape[1],d))
    feats = apply_Gauss_norm(feats)
    for i,arr in enumerate(feats):
        feats[i] = np.dot(arr, proj)
    return feats


def cos_sim(x, z):
    """x,z are 2d time series arrays, t x d"""

    dot_product = np.diagonal(np.matmul(x,z.T))
    norms = 1e-9 +  np.linalg.norm(x,axis=1) * np.linalg.norm(z,axis=1) 
 
    return dot_product / norms


def joint_distances(shL, shR, elL, elR, wrL, wrR, neck):
    len_shoulder = np.linalg.norm((shL - shR), axis=1)
    # avoid 0 division, replace too smalls with mean
    len_shoulder[len_shoulder < len_shoulder.mean() * .3] = len_shoulder.mean()

    len_forearm_R = np.linalg.norm((elR - wrR), axis=1) / len_shoulder
    len_forearm_L = np.linalg.norm((elL - wrL), axis=1) / len_shoulder
    len_up_arm_R = np.linalg.norm((elR - shR), axis=1) / len_shoulder
    len_up_arm_L = np.linalg.norm((elL - shL), axis=1) / len_shoulder
    hands_dist = np.linalg.norm((wrL - wrR), axis=1) / len_shoulder

    hand_positionR = (wrR - neck) / len_shoulder[:, None]
    hand_positionL = (wrL - neck) / len_shoulder[:, None]

    return np.vstack((len_shoulder, len_forearm_L, len_forearm_R, len_up_arm_L, len_up_arm_R,
                      hands_dist, hand_positionL.T, hand_positionR.T))


def joint_cos_sim(shL, shR, elL, elR, wrL, wrR, neck):
    " measure the angles in terms of cos sim "

    elbow_angle_L = cos_sim(wrL - elL, shL - elL)
    elbow_angle_R = cos_sim(wrR - elR, shR - elR)

    shoulder_angle_L = cos_sim(elL - shL, shR - shL)
    shoulder_angle_R = cos_sim(elR - shR, shL - shR)

    neck_angle = cos_sim(neck - (shL + shR) / 2, shL - shR)

    return np.vstack((elbow_angle_L, elbow_angle_R, shoulder_angle_L, shoulder_angle_R, neck_angle))


def velocity_dx_dy(seq, d=1):
    # same pad at the end
    displacemnt = np.zeros_like(seq)
    displacemnt[:-d] = seq[d:] - seq[:-d]
    displacemnt[-d:] = displacemnt[-d-1]

    return displacemnt


def velocity_vector(seq, d=1):
    # seq is t x 2, find displacement vectors cos sim with horizontal and its magnitude 

    displacemnt = velocity_dx_dy(seq, d=d)
    
    unit_vec = np.ones_like(seq)
    unit_vec[:,1] = 0

    angle = cos_sim(displacemnt, unit_vec)
    magnitude = np.linalg.norm(displacemnt, axis=1)

    return np.vstack((angle, magnitude))


def openpose25_joints(feats_arr):
    
    # center wrt neck mean    
    neck = feats_arr[:,0,:2]    
    origin = neck.mean(0)
    
    # normalize wrt shoulders' dist.
    shL = (feats_arr[:,5,:2] - origin)  #shoulders
    shR = (feats_arr[:,2,:2] - origin) 
    len_shoulder = np.linalg.norm((shL - shR), axis=1).mean()
    
    shL = shL / len_shoulder
    shR = shR / len_shoulder    
    elL = (feats_arr[:,6,:2] - origin) / len_shoulder #elbows
    elR = (feats_arr[:,3,:2] - origin) / len_shoulder
    wrL = (feats_arr[:,7,:2] - origin) / len_shoulder #wrists
    wrR = (feats_arr[:,4,:2] - origin) / len_shoulder
    neck = (feats_arr[:,0,:2] - origin) / len_shoulder
        
    return shL, shR, elL, elR, wrL, wrR, neck

def alphapose17_joints(feats_arr):
    
    shL = feats_arr[:,5,:2] #shoulders
    shR = feats_arr[:,6,:2]
    wrL = feats_arr[:,9,:2] #wrists
    wrR = feats_arr[:,10,:2]
    elL = feats_arr[:,7,:2] #elbows
    elR = feats_arr[:,8,:2]
    neck = feats_arr[:,0,:2]
    
    return shL, shR, elL, elR, wrL, wrR, neck


def derive_joint_feaures(shL, shR, elL, elR, wrL, wrR, neck):
    angles = joint_cos_sim(shL, shR, elL, elR, wrL, wrR, neck)
    lengths = joint_distances(shL, shR, elL, elR, wrL, wrR, neck)
    velocityL = velocity_vector(wrL) / lengths[0]  # normalize with shoulder dist
    velocityR = velocity_vector(wrR) / lengths[0]

    # dont take the first length, its shoulder distance
    derived_features = np.vstack((angles, lengths[1:], velocityL, velocityR))

    return np.asanyarray(derived_features).T


def prepare_alphapose(feat3d):

    shL, shR, elL, elR, wrL, wrR, neck = alphapose17_joints(feat3d)

    derived_features = derive_joint_feaures(shL, shR, elL, elR, wrL, wrR, neck)

    return derived_features


def derive_openpose25_gaussian(feats_arr):
    shL, shR, elL, elR, wrL, wrR, neck = openpose25_joints(feats_arr)
    
    wrR_vel = velocity_dx_dy(wrR)
    wrL_vel = velocity_dx_dy(wrL)
    elR_vel = velocity_dx_dy(elR)
    elL_vel = velocity_dx_dy(elL)
    
    combined_feats = np.hstack((wrR, wrL, elR, elL, wrR_vel, wrL_vel, elR_vel, elL_vel))
    
    return combined_feats


def derive_op25_joint_dist_(feats_arr):
    shL, shR, elL, elR, wrL, wrR, neck = openpose25_joints(feats_arr)
    
    lengths = joint_distances(shL, shR, elL, elR, wrL, wrR, neck)[1:].T
    wrR_vel = velocity_dx_dy(wrR)
    wrL_vel = velocity_dx_dy(wrL)
    
    return np.hstack((lengths, wrR_vel, wrL_vel))


def derive_openpose(feat3d):

    shL, shR, elL, elR, wrL, wrR, neck = openpose25_joints(feat3d)

    derived_features = derive_joint_feaures(shL, shR, elL, elR, wrL, wrR, neck)

    return derived_features


def center_body(feats_arr):
    """ select only upper body keypoints and take neck as origin"""

    selected_body_kp = np.concatenate((np.arange(0, 9), np.arange(15, 19)))
    return feats_arr[:, selected_body_kp, :2] - feats_arr[:, 1, :2].mean(axis=0)


def center_face(feats_arr):
    """ take bottom chin as origin"""
    return feats_arr[:, :, :2] - feats_arr[:, 8, :2].mean(axis=0)

def center_hand(feats_arr):
    """ take bottom chin as origin??? bilek olmasin??"""
    return feats_arr[:, :, :2] - feats_arr[:, 0, :2].mean(axis=0)


def prepare_openpose(path, seq_name):

    feat = np.load(join(path, seq_name + '.npy'))

    if 'body' in path:
        feat = center_body(feat)
    if 'face' in path:
        feat = center_face(feat)
    if 'hand' in path:
        feat = center_hand(feat)
#        feat = feat[:,:,:2]
    feat = feat.reshape(feat.shape[0],-1)

    return feat


def get_features_array_for_seq_name(seq_name, feature_paths, feature_type):
    """ feature_paths : list  """

    for i,path in enumerate(feature_paths):

#        print(seq_name)
        if feature_type == 'open-pose':
            feat = prepare_openpose(path, seq_name)
        else:
            feat = np.load(join(path, seq_name + '.npy'))
            if 'alpha' in path:
                feat = prepare_alphapose(feat)

#        print(feat.shape)
        if i == 0:
            array2d = feat
        else:
            array2d = np.concatenate((array2d,feat),axis=1)

    return array2d


def get_paths_for_deephand_features(root_dir, features):

    feature_paths = []

    deephand = 'deep_hand'
#    deephand = 'deep_hand_original_crop'

    for feat in features['feats']:
        for hand in features['hands']:
            feature_paths.append( join( root_dir,deephand,feat,hand,'train'))

    return feature_paths


def get_paths_for_openpose_features(root_dir, features):

    feature_paths = []

    for feat in features['feats']:
#            feature_paths.append( join( root_dir,feat,'train'))
            feature_paths.append( join( root_dir,feat))

    return feature_paths


def get_paths_for_vgg_features(root_dir, features):

    feature_paths = []

    for feat in features['feats']:
            feature_paths.append( join( root_dir,feat))

    return feature_paths


def get_paths_for_alphapose_features(root_dir, features):

    feature_paths = []

    feature_paths.append( join( root_dir,'alphapose','train'))

    return feature_paths


def get_paths_for_features(feats_root, features, feature_types):

    feature_paths = []

    for i,feature_type in enumerate(feature_types):

        if 'deep-hand' in feature_type :
            feature_paths.extend( get_paths_for_deephand_features(feats_root[i], features[i]) )
        if 'open-pose' in feature_type:
            feature_paths.extend( get_paths_for_openpose_features(feats_root[i], features[i]) )
        if 'vgg' in feature_type:
            feature_paths.extend( get_paths_for_vgg_features(feats_root[i], features[i]) )
        if 'alpha' in feature_type:
            feature_paths.extend( get_paths_for_alphapose_features(feats_root[i], features[i]) )

    return feature_paths


"""
def get_feats_for_foldername(img_root,folder,img_idx):

    paths = listdir( join( img_root, folder, '1'))

    img = plt.imread(join( img_root, folder, '1', paths[img_idx]))

    return img
"""


