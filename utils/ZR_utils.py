import numpy as np
from os.path import join, isfile
from os import chdir, makedirs, remove
import pandas as pd
from utils.db_utils import img_paths_for_folder, fragment_tokenizer, gold_fragments_df_for_signer, dominant_label
import matplotlib.pyplot as plt
from shutil import rmtree, copyfile
import subprocess
import glob
from utils.feature_utils import apply_Gauss_norm
import os

zr_root = '/home/korhan/Desktop/zerospeech2017/track2/src/ZRTools'
lsh_path = '/home/korhan/Dropbox/tez/features/to_zrtools/lsh/tmp'
scriptsroot = '/home/korhan/Dropbox/tez_scripts'



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


from utils.feature_utils import get_features_array_for_seq_name, get_paths_for_features
from shutil import rmtree
from os import mkdir


def prepare_inputs(exp_name, seq_names, lsh_path, feats_root, features, feature_type, softmax=False, d=61, normalize=True):

    file_list_name = join(lsh_path, '..', '..', exp_name + '.lsh64.lst')

    rmtree(lsh_path, ignore_errors=True)
    mkdir(lsh_path)

    feature_paths = get_paths_for_features(feats_root, features, feature_type)

    text_file = open(file_list_name, 'w')

    for seq_name in seq_names:
        array2d = get_features_array_for_seq_name(seq_name, feature_paths, feature_type)

        if d == 60:
            array2d = np.hstack([array2d[:,1:d+1],array2d[:,d+2:2*d+2], array2d[:,2*d+3:3*d+3]])

        if softmax:

            array2d[:,:d] = np.exp(array2d[:,:d]) / np.exp(array2d[:,:d]).sum(axis=1)[:,None]
            array2d[:,d:2*d] = np.exp(array2d[:,d:2*d]) / np.exp(array2d[:,d:2*d]).sum(axis=1)[:,None]
            array2d[:,2*d:] = np.exp(array2d[:,2*d:]) / np.exp(array2d[:,2*d:]).sum(axis=1)[:,None]


        sigs = convert_npy_to_lsh(array2d, normalize)

        sigs.tofile(join(lsh_path, seq_name + '.std.lsh64'))

        text_file.writelines(join(lsh_path, seq_name + '.std.lsh64\n'))

#        print('lsh signatures for {} are saved..'.format(seq_name))

    text_file.close()

    return file_list_name


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


#def get_projmat(D):    return np.float32(np.random.randn(64*D))


def convert_2d_to_1d(array2d, dtype=np.float32):
    n_frames, n_dim = array2d.shape
    return array2d.reshape((n_dim * n_frames)).astype(dtype=dtype)


def std_to_lsh64(D, feat_std, projmat):
    n_frames = int(len(feat_std) / D)
    sigs = np.zeros((n_frames*2),dtype=np.uint32)
    S = 32
    for i in range(n_frames):

        for b in range(S):
            proj = np.dot(feat_std[i*D:(i+1)*D] , projmat[b*D:(b+1)*D])
            sigs[2*i] += (proj>0) * 2**b

        for b in range(S):
            proj = np.dot( feat_std[i*D:(i+1)*D ] , projmat[(b+S)*D : (b+S+1)*D ] )
            sigs[2*i + 1] += (proj>0) * 2**b

    return sigs


def convert_npy_to_lsh(array2d):

    n_frames, n_dim = array2d.shape

    array1d = convert_2d_to_1d(array2d, dtype=np.float32)

#    if n_dim == 64: projmat = np.eye(n_dim, dtype=np.float32).reshape(-1)
#    else: 
    projmat = get_projmat(D=n_dim)

    sigs = std_to_lsh64(D=n_dim, feat_std=array1d, projmat=projmat)

    return sigs


def prepare_inputs2(exp_name, seq_names, feats, lsh_path, normalize=True):

    if lsh_path[-len(exp_name):] != exp_name: lsh_path = join(lsh_path,exp_name)

    file_list_name = join(lsh_path, '..', '..', exp_name + '.lsh64.lst')

    if normalize and (feats[0] is not None): feats = apply_Gauss_norm(feats)

    # rmtree(lsh_path, ignore_errors=True)
    os.makedirs(lsh_path, exist_ok=True)

    text_file = open(file_list_name, 'w')

    for i,seq_name in enumerate(seq_names):

        if os.path.isfile( join(lsh_path, seq_name + '.std.lsh64') ): continue

        array2d = feats[i]

        sigs = convert_npy_to_lsh(array2d)

        sigs.tofile(join(lsh_path, seq_name + '.std.lsh64'))

        text_file.writelines(join(lsh_path, seq_name + '.std.lsh64\n'))

#        print('lsh signatures for {} are saved..'.format(seq_name))

    text_file.close()

    return file_list_name



def run_disc_2(filelist, sge, zr_root):

    chdir(scriptsroot)

    command = './run_zr17.sh {} {} {}'.format(filelist, sge, zr_root)

    print(command)

    subprocess.call(command.split())



def run_disc(filelist):

    # import subprocess

    chdir(zr_root)

    command = './run_disc nosge {}'.format(filelist)

    print(command)

    subprocess.call(command.split())


def change_plebdisc_thr(rhothr=0, T=0.25, D=5, dx=25, dy=5, medthr=0.5, Tscore=0.5, R=10, castthr=7, trimthr=0.25, B=50, P=8 ):

    with open(join(zr_root, 'scripts/plebdisc_filepair')) as f:
        lines = f.readlines()

    assert lines[29][:17] == 'plebdisc/plebdisc'

    # remove(join(zr_root, 'scripts/plebdisc_filepair'))

    suffix = ' -file1 $LSH1 -file2 $LSH2 | awk \'NF == 2 || $5 > 0. {print $0;}\' > $TMP/${BASE1}_$BASE2.match\n'

    new_cmd = 'plebdisc/plebdisc -S 64 -P {} -rhothr {} -T {} -B {} -D {} -dtwscore 1 \
                -kws 0 -dx {} -dy {} -medthr {} -twopass 1 -maxframes 90000 -Tscore {} -R {} -castthr {} -trimthr {}'.format(
                P, rhothr, T, B, D, dx, dy, medthr, Tscore, R, castthr, trimthr)

    lines[29] = new_cmd + suffix

    with open(join(zr_root, 'scripts/plebdisc_filepair'), 'w') as file:
        file.writelines(lines)



def change_post_disc_thr(olapthr=0.9, dedupthr=0.2, durthr=5, rhothr=1000, min_edge_w=0):

    with open(join(zr_root, 'post_disc')) as f:
        lines = f.readlines()

    lineid = 25

    assert lines[lineid][:7] == 'OLAPTHR'

    # remove(join(zr_root, 'post_disc'))

    lines[lineid:lineid+5] = ['OLAPTHR={}\n'.format(olapthr),
                   'DEDUPTHR={}\n'.format(dedupthr),
                   'DURTHR={}\n'.format(durthr),
                   'RHOTHR={}\n'.format(rhothr),
                    'THRESH={}\n'.format(min_edge_w)]

    with open(join(zr_root, 'post_disc'), 'w') as file:
        file.writelines(lines)


def full_exp_path(exp_):
    exp_root = '/home/korhan/Desktop/zerospeech2017/track2/src/ZRTools/exp/'

    try:
        name_root = glob.glob(exp_root + 'zrroot0_{}*/exp/*'.format(exp_))[0]
        exp_name = '/'.join(name_root.split('/')[-3:])[:-6]
    except:
        name_root = glob.glob(exp_root + '{}*/'.format(exp_))[0]
        exp_name = glob.glob(exp_root + '{}*/'.format(exp_))[0].split('/')[-2][:-6]

    return name_root, exp_name



def post_disc(exp_name, dtw_thr):

    chdir(zr_root)

    command = './post_disc {} {}'.format(exp_name+'.lsh64', dtw_thr)

    print(command)

    subprocess.call(command.split())

    results_path = join(zr_root, 'exp', exp_name + '.lsh64', 'results')

    return results_path


def post_disc2(expdir, exp_name, dtw_thr):

    chdir(scriptsroot)

    command = './postdisc_zr17.sh {} {} {} {}'.format(zr_root, exp_name, dtw_thr, expdir)

    subprocess.call(command.split())

    results_path = join(expdir, exp_name, 'results')

    return results_path


""" MATCH DISPLAY """

def new_match_dict(last_filepair, new_line):
    match = {'f1': last_filepair[0],
             'f2': last_filepair[1],
             'f1_start': int(new_line[0]),
             'f1_end': int(new_line[1]),
             'f2_start': int(new_line[2]),
             'f2_end': int(new_line[3]),
             'score': float(new_line[4])
            }

    if len(new_line) == 6:
        match['rho'] = float(new_line[5])

    return match


def read_matches_outfile(match_file_path):

    min_frame_th = 5

    matches_list = []

    match_file = open(match_file_path, 'r')

    last_filepair = match_file.readline().strip('\n').split(' ')
#    print((last_filepair))

    for i, line in enumerate(match_file):

        new_line = line.strip('\n').split(' ')

        if len(new_line) == 2:
            last_filepair = new_line

        elif len(new_line) == 6:
            if ( (int(new_line[1])-int(new_line[0])) >= min_frame_th ) & ( (int(new_line[3])-int(new_line[2])) >= min_frame_th ):
                matches_list.append(new_match_dict(last_filepair, new_line))

        else:
            print('ERROR: unexpected line: {}'.format(new_line))

    #    if i >9: break
    match_file.close()

    return matches_list


def get_matches_df(match_file_path):
    # type: (str) -> pd.DataFrame

    if 'out' not in match_file_path:     # if exp_name is given
        match_file_path = join(zr_root, 'exp', match_file_path + '.lsh64', 'matches', 'out.1')

    matches_list = read_matches_outfile(match_file_path)

    matches_df = pd.DataFrame.from_records(matches_list,
                                           columns=['f1', 'f2', 'f1_start', 'f1_end', 'f2_start', 'f2_end', 'score', 'rho']
                                           )

    matches_df = matches_df.astype(dtype={'f1': str, 'f2': str,
                                          'f1_start': int, 'f1_end': int, 'f2_start': int, 'f2_end': int,
                                          'score': float, 'rho': float}
                                   )  # type: pd.DataFrame
#    print(matches_df.head(3))
    print('Read {} matches'.format(len(matches_df)))
    if len(matches_df) > 100000:
        matches_df = matches_df.sort_values(by='score', ascending=False)[:50000].reset_index(drop=True)

    return matches_df


def get_matches_all(exp_name, exp_root='/home/korhan/Desktop/zerospeech2017/track2/src/ZRTools/exp/'):

    if 'matches' not in exp_name:
        outdir_m = os.path.join(exp_root, '{}.lsh64/matches'.format(exp_name) )
    else: outdir_m = exp_name

    outfiles = sorted(glob.glob(outdir_m + '/out.*'))
    if (len(outfiles)) == 0: 
        print('DIRECTORY ERROR', outdir_m)

        return 0
        
    out_df_list = []
    for name in outfiles:
        out_df_list.append( get_matches_df(name) )

        
    # append all matches df to a single df
    matches_df = pd.concat(out_df_list, ignore_index=True)

    return matches_df



def frame_info(row, file_i):
    folder_name = row['f{}'.format(file_i)]
    start = row['f{}_start'.format(file_i)]
    end = row['f{}_end'.format(file_i)]

    return folder_name, start, end


def match_row_to_img_paths(img_root, row):
    img_paths = dict()

    for file_i in ['1', '2']:
        folder_name, start, end = frame_info(row, file_i)

        img_paths[file_i] = img_paths_for_folder(img_root, folder_name, start, end+1)

    return img_paths


def match_row_to_labels(df, row):
    label_ids = dict()
    label_names = dict()

    for file_i in ['1', '2']:
        folder_name, start, end = frame_info(row, file_i)

        file_df = df[df.folder == folder_name][start: end + 1]  # assume frames start from 0
        label_ids[file_i] = list(file_df.label)
        label_names[file_i] = list(file_df.label_name)

    return label_ids, label_names


def match_row_to_annot(row, annotation_df):
    annots = dict()

    for file_i in ['1', '2']:
        folder_name, start, end = frame_info(row, file_i)

        annots[file_i] = annotation_df.annotation[annotation_df.id == folder_name].item()

    return annots


def discovered_fragments(matches_df):
    F_disc = pd.DataFrame(columns=['filename', 'start', 'end'])
    for i, row in matches_df.iterrows():
        F_disc = F_disc.append([{'filename': row['f1'],
                                 'start': row['f1_start'],
                                 'end': row['f1_end']}], ignore_index=True)
        F_disc = F_disc.append([{'filename': row['f2'],
                                 'start': row['f2_start'],
                                 'end': row['f2_end']}], ignore_index=True)

    return F_disc


def plot_match_stats(matches_df):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 4))
    axes[0].hist(matches_df.score)
    axes[0].set_title('DTW scores')
    axes[1].hist(pd.concat([(matches_df.f2_end - matches_df.f2_start), (matches_df.f1_end - matches_df.f1_start)]),
                 bins=range(50))
    axes[1].set_title('fragment lengths distribution')
    axes[2].hist(matches_df.rho)
    axes[2].set_title('rho scores')


""" POST DISC DISPLAYING """

def new_node_dict(new_line):
    match = {'filename': str(new_line[0]),
             'start': int(new_line[1]),
             'end': int(new_line[2]),
             'score': float(new_line[3]),
             'unknown': float(new_line[4]),
             'file_id': int(new_line[5])
            }

    return match


def get_nodes_list(results_path):

    nodes_list = []

    nodes_file = open(results_path + '/master_graph.nodes', 'r')

    for i, line in enumerate(nodes_file):

        new_line = line.strip('\n').split('\t')

        nodes_list.append(new_node_dict(new_line))

    nodes_file.close()

    print('Read {} nodes/tokens'.format(len(nodes_list)))

    return nodes_list


def get_nodes_df(results_path):

    if results_path[-7:] != 'results':     # if exp_name is given
        results_path = join(zr_root, 'exp', results_path + '.lsh64', 'results')

    nodes_list = get_nodes_list(results_path)

    nodes_df = pd.DataFrame.from_records(nodes_list,
                                         columns=['filename', 'start', 'end', 'score', 'unknown', 'file_id']
                                         )

    nodes_df.index += 1

    return nodes_df


def get_clusters_list(results_path):

    if results_path[-7:] != 'results':     # if exp_name is given
        results_path = join(zr_root, 'exp', results_path + '.lsh64', 'results')

    clusters_list = []

    cluster_file = open(results_path + '/master_graph.dedups', 'r')

    for i, line in enumerate(cluster_file):
    #    print(line)
        new_line = line.strip('\n').split(' ')

        clusters_list.append([int(i) for i in new_line])

    cluster_file.close()

    print('Read {} clusters'.format(len(clusters_list)))

    return clusters_list


def write_lsh_filelist(seq_names, params):
    # prepare filelist, select from seqnames
    lshfilelist = join(params['exp_root'], params['expname'] + '.lst')
    with open(lshfilelist, 'w') as f:
        f.write('\n'.join([glob.glob(join(
                params['lshroot'],params['featype'], name + '.std.lsh64' ))[0] for name in seq_names]))
        
    return lshfilelist


def change_nlines_per_job(n, params):
    with open(join(params['zrroot'], 'config')) as f:
        lines = f.readlines()

    lineid = 41
    
    assert lines[lineid][:7] == 'NJ_DISC'

    lines[lineid] = 'NJ_DISC={}\n'.format(int(n))

    with open(join(params['zrroot'], 'config'), 'w') as file:
        file.writelines(lines)


def change_njobs_for_nseq(n, params):
    # change config file according to len(seq_names)
    max_job = 5000
    nslot = 6
    n_required = (n*(n-1)/2 + n) 

    while n_required / nslot > max_job: nslot += nslot
    set_n = int(n_required / nslot + 1)        

    change_nlines_per_job(set_n, params)
    

def change_params(p):

    change_plebdisc_thr(rhothr=p['rhothr'], T=p['T'], D=p['D'], dx=p['dx'], dy=p['dy'], 
                        medthr=p['medthr'] , Tscore=p['Tscore'] , R=p['R'] , 
                        castthr=p['castthr'] , trimthr=p['trimthr']  )


def run_disc_ZR(feats_dict, params):
    
    seq_names = sorted(feats_dict.keys())
    
    # if not os.path.exists(join(params['lshroot'], params['featype'])):
    print('Generating lsh files ')        
    lshfileslst = prepare_inputs2(params['featype'], feats_dict.keys(), 
                                  list(feats_dict.values()), params['lshroot'], normalize=True)

    filelist = write_lsh_filelist(seq_names, params)

    change_njobs_for_nseq(len(seq_names), params)

    change_params(params['disc'])
    # rundisc
    run_disc_2(filelist=filelist, sge='sge', zr_root=params['zrroot'])
    # get matches df
    matches_df = get_matches_all( join(params['exp_root'], params['expname'] + '/matches'))

    matches_df['cost'] = 1 - matches_df['score']
    
    return matches_df


