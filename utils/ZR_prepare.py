import numpy as np
from os.path import join, isfile
from os import chdir, makedirs, remove
from shutil import rmtree, copyfile
from os import mkdir


zr_root = ''

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



def gen_proj(D, dest='/home/korhan/Dropbox/tez/features/to_zrtools/projmats'):
    ''' calls zr-genproj to generate a projection matrix for given dims
    :returns full path of the projmat'''

    import subprocess

    command = '{}/plebdisc/genproj -D {} -S 64 -seed 1 -projfile {}/proj_S64xD{}_seed1'.format(zr_root,D,dest,D)

    subprocess.call(command.split())

    return '{}/proj_S64xD{}_seed1'.format(dest,D)


def get_projmat(D, dest='/home/korhan/Dropbox/tez/features/to_zrtools/projmats'):

    path = '{}/proj_S64xD{}_seed1'.format(dest, D)

    if not isfile(path):
        path = gen_proj(D, dest)

    mat = np.fromfile(path, dtype=np.float32)

    return mat



def convert_2d_to_1d(array2d, dtype=np.float32):
    n_frames, n_dim = array2d.shape
    return array2d.reshape((n_dim * n_frames)).astype(dtype=dtype)


def std_to_lsh64(D, feat_std, projmat,muth=0):
    n_frames = int(len(feat_std) / D)
    sigs = np.zeros((n_frames*2),dtype=np.uint32)
    S = 32
    for i in range(n_frames):

        for b in range(S):
            proj = np.dot(feat_std[i*D:(i+1)*D] , projmat[b*D:(b+1)*D])
            sigs[2*i] += (proj>muth) * 2**b

        for b in range(S):
            proj = np.dot( feat_std[i*D:(i+1)*D ] , projmat[(b+S)*D : (b+S+1)*D ] )
            sigs[2*i + 1] += (proj>muth) * 2**b

    return sigs


def convert_npy_to_lsh(array2d, muth=0):

    n_frames, n_dim = array2d.shape

    array1d = convert_2d_to_1d(array2d, dtype=np.float32)

#    if n_dim == 64: projmat = np.eye(n_dim, dtype=np.float32).reshape(-1)
#    else: 
    projmat = get_projmat(D=n_dim)

    sigs = std_to_lsh64(D=n_dim, feat_std=array1d, projmat=projmat, muth=muth)

    return sigs


def prepare_inputs(exp_name, seq_names, feats, lsh_path, normalize=True, muth=0):
    """
    given feature arrays and corresponding filenames, prepare lsh bit
    signatures and save them to lsh_path
    params:
        exp_name (str): experiment name 
        seq_names (list of strings): sequence names
        feats (list of [txd] np arrays): feats and seq_names should be in the same order
        lsh_path (str): save dir
    """

    if lsh_path[-len(exp_name):] != exp_name: lsh_path = join(lsh_path,exp_name)

    file_list_name = join(lsh_path, '..', '..', exp_name + '.lsh64.lst')

    if normalize: feats = apply_Gauss_norm(feats)

    rmtree(lsh_path, ignore_errors=True)
    mkdir(lsh_path)

    text_file = open(file_list_name, 'w')

    for i,seq_name in enumerate(seq_names):
        array2d = feats[i]

        sigs = convert_npy_to_lsh(array2d, muth)

        sigs.tofile(join(lsh_path, seq_name + '.std.lsh64'))

        text_file.writelines(join(lsh_path, seq_name + '.std.lsh64\n'))

#        print('lsh signatures for {} are saved..'.format(seq_name))

    text_file.close()

    return file_list_name
