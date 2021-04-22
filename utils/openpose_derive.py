import numpy as np
from os.path import join
import itertools





def get_body(feats_root, seq_name):

    feats_path = join(feats_root,'op_body', seq_name + '.npy')
    feats_body = np.load(feats_path)

    # discard lower body
    feats_body = feats_body[:,np.concatenate((np.arange(0, 8), np.arange(15, 19))),:]

    # normalize offset
    feats_body = feats_body[:,:,:2] - np.expand_dims(feats_body[:,1,:2],1)

    # normalize wrt shoulder dist
    shoulder_L = np.linalg.norm(feats_body[:,5]-feats_body[:,2],2,1).mean()
    feats_body = feats_body / shoulder_L

    return feats_body, shoulder_L


def get_hand(feats_root, rl, seq_name, normalizng_scale):
    feats_path = join(feats_root,'op_hand', rl, seq_name + '.npy')
    feats_hand_R = np.load(feats_path)

    # normalize offset
    feats_hand_R = feats_hand_R[:,:,:2] - np.expand_dims(feats_hand_R[:,0,:2],1)
    # normalize wrt shoulder dist
    feats_hand_R = feats_hand_R / normalizng_scale
    
    return feats_hand_R


# ji -> [t,2] joint coordinate series
def joint_joint_distance(j1, j2):
    # euclidean distance between two time series
    return np.sqrt(np.sum((j1-j2)**2,1))


def joint_joint_orientation(j1, j2):
    # orientation unit vector from j1 to j2
    return (j2-j1) / (np.linalg.norm(j2-j1,2,1)[:,None]+ 1e-9)


def line_line_angle(line1, line2):
    # angle between j1->j2 and j3->j4
    j1,j2 = line1[:,0,:], line1[:,1,:]
    j3,j4 = line2[:,0,:], line2[:,1,:]
    elementwise_product = np.multiply(joint_joint_orientation(j1,j2), joint_joint_orientation(j3,j4))
    return np.arccos(np.sum(elementwise_product,1 ))


def joint_plane_distance(j0, plane):
    j1,j2,j3 = plane
    
    
def joint_combinations(arr_shape, c=2):
    # return all possible pairs
    lines = []
    for comb in itertools.combinations(range(arr_shape), c):
        lines.append(comb)
    
    return np.array(lines)


def distances_pairs(arr, lines):
    # return [t,d] 
    distances = np.zeros((arr.shape[0],len(lines)))
    for i in range(len(lines)):
        distances[:,i] = joint_joint_distance(arr[:,lines[i,0]], arr[:,lines[i,1]] )
        
    return distances


def orients_pairs(arr, lines):
    orients = np.zeros((arr.shape[0],2,len(lines)))
    for i in range(len(lines)):
        orients[:,:,i] = joint_joint_orientation(arr[:,lines[i,0]], arr[:,lines[i,1]] )

    return orients.reshape(orients.shape[0],-1)

def angles_planes(arr, planes):
    angles = np.zeros((arr.shape[0],len(planes)))
    for i in range(len(planes)):
        angles[:,i] = line_line_angle(arr[:,planes[i,:2]], arr[:,planes[i,-2:]] )

    return angles


def static_features(feat_arr):
    
    lines = joint_combinations(feat_arr.shape[1], 2)
    planes = joint_combinations(feat_arr.shape[1], 3)

    distances = distances_pairs(feat_arr, lines)
    orients = orients_pairs(feat_arr, lines)
    angles = angles_planes(feat_arr, planes)

    return np.concatenate([distances, orients, angles],1)