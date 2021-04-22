import numpy as np
import os
import sys
import glob
import pandas as pd
from zipfile import ZipFile
import shutil
import subprocess
import pickle


class BEER:
    def __init__(self, feats_dict, params):
        self.uttids = list(feats_dict.keys())
        # self.feats = feats
        self.feats_dict = feats_dict
        self.pr = params
        self.root_dir = params['rootdir']

        self.data_unit_path = os.path.join(self.root_dir, 'data', self.pr['db'], self.pr['expname'])

        self.features_unit_path = os.path.join(self.root_dir, 'features',  self.pr['db'])

        self.outdir = os.path.join('exp',self.pr['db'],self.pr['subset'], self.pr['expname'] )

        self.modelpth = os.path.join('exp', self.pr['db'], 'datasets',
                                     self.pr['feaname'],self.pr['expname']+'.pkl ')

        self.__prepared = False


    def clean_dirs(self):
        for path in [self.data_unit_path, self.outdir, self.modelpth ]:
            if os.path.exists(path): shutil.rmtree(path)



    def save_uttids(self):
        # save filenames as 'uttids'
        os.makedirs(self.data_unit_path, exist_ok=True)

        print('Writing filenames to {}'.format(os.path.join(self.data_unit_path,'uttids')) )

        with open(os.path.join(self.data_unit_path,'uttids'),'w') as f:
            for name in self.uttids:
                f.writelines(name + '\n')
                
    def save_units(self):
        # save target unit IDs as 'units'
        os.makedirs(self.data_unit_path, exist_ok=True)

        print('Writing {} target units to {}'.format(self.pr['nunits'], os.path.join(self.data_unit_path, 'units')) )

        with open(os.path.join(self.data_unit_path, 'units'),'w') as f:
            f.writelines('sil non-speech-unit\n')
            for i in range(1,self.pr['nunits']+1):
                f.writelines('au{} speech-unit\n'.format(i))


    def save_npy_files(self):
        # save npy files                   
        if os.path.exists(self.features_unit_path):
            shutil.rmtree(self.features_unit_path)

        os.makedirs(self.features_unit_path, exist_ok=True)
        os.makedirs(os.path.join(self.features_unit_path, self.pr['feaname']), exist_ok=True)

        for key, value in self.feats_dict.items():
            np.save(os.path.join(self.features_unit_path, self.pr['feaname'], key), value)


    def archive_to_zip(self):
        # arcihve to a single npz file

        print('Saving features to {}'.format(self.features_unit_path + '/'+ self.pr['feaname'] + '.npz'))

        with ZipFile(self.features_unit_path + '/'+ self.pr['feaname'] + '.npz', 'w') as f:
            for path in glob.glob(self.features_unit_path + '/' + self.pr['feaname'] + '/*.npy'):
                arcname = os.path.basename(path).replace('.npy' , '')
                f.write(path, arcname=arcname)


    def get_commands(self):
                 
        cmd_create_db = 'steps/create_dataset.sh {}  {}/{}.npz  {}'.format(
                    self.data_unit_path, 
                    self.features_unit_path, self.pr['feaname'], 
                    self.modelpth)

        cmd_aud = ('steps/aud.sh --prior {} --acoustic-scale {} --parallel-opts {} --parallel-njobs {} '.format(
                    self.pr['prior'], self.pr['ac_scale'], self.pr['opt_prll'], self.pr['njobs']) + 
                    ' '.join([
                    self.pr['conf'],
                    self.data_unit_path,
                    self.data_unit_path,
                    self.modelpth,
                    str(self.pr['epochs']),
                    self.outdir
                    ]) )

        cmd_decode = ('steps/decode.sh ' +
                    '--per-frame --parallel-opts {} --parallel-njobs {} '.format(
                    self.pr['opt_prll'], self.pr['njobs']) +
                    self.outdir + '/final.mdl ' + 
                    self.data_unit_path + '/uttids ' +
                    self.modelpth + 
                    self.outdir + '/decode_perframe/')
        
        return cmd_create_db, cmd_aud, cmd_decode


    def prepare(self, prints=True):

        self.save_uttids()
        self.save_units()
        self.save_npy_files()
        self.archive_to_zip()
#         self.create_outdir()
        self.cmds = self.get_commands()

        self.__prepared = True

        if prints: 
            for cmd in self.cmds: print(cmd + '\n')
            
        # return cmds


    def write_disc_script(self, expname=''):

        if not self.__prepared: self.prepare(prints=False)

        with open(os.path.join(self.root_dir, 'run_discovery_{}.sh'.format(expname)), 'w') as f:
            f.writelines('#!/usr/bin/env bash\n\nset -e\n\n. path.sh\n\n')
            f.write('rm -rf {}'.format(self.outdir + '/decode_perframe/'))
            for cmd in self.cmds: f.write('\n\n' + cmd)


    def run_disc(self, expname=''):
        
        print('running commands')
        cmds = ['chmod +x run_discovery_{}.sh'.format(expname),
                './run_discovery_{}.sh'.format(expname)]

        os.chdir(self.pr['rootdir'])
        subprocess.call('pwd')

        for cmd in cmds:
            subprocess.call(cmd, shell=True)


    def prepare_and_run_disc(self, expname='', clean=False):

        if clean: self.clean_dirs()

        self.write_disc_script(expname)
        self.run_disc(expname)

    ''' post discovery '''


    def read_transcription(self):
        trans_file = os.path.join(self.root_dir, self.outdir, 'decode_perframe', 'trans')

        with open(trans_file, 'r') as f:
            lines = f.readlines()

        return lines


    def read_elbos(self):
        trainlog = os.path.join(self.root_dir, self.outdir, 'training.log')

        lines = []
        with open(trainlog, 'r') as f:
            for line in f.readlines():
                lines.append(float(line.strip('INFO: accumulated ELBO=\n')))

        return lines      
    
    
    def get_transcription_whole(self):
        # get all transcription as concat
        lines = self.read_transcription()
        res = []
        for line in lines:
            res.extend(line.strip('\n').split(' ')[1:])
        
        return res


    @staticmethod
    def read_to_dict(lines):
        # read them into a dict
        word_clusters = set()
        trans_dict = dict()
        for line in lines:
            filename = line.strip('\n').split(' ')[0]
            trans = line.strip('\n').split(' ')[1:]
            trans_dict[filename] = trans
            word_clusters |= set(trans)

        return trans_dict, word_clusters


    def get_transcriptions_dict(self):

        lines = self.read_transcription()
        self.trans_dict = self.read_to_dict(lines)

        return self.trans_dict


    @staticmethod
    def get_nodes(trans_dict):
        fragments = []

        # for filename in self.uttids:
        for filename, trans in trans_dict.items():
            # trans = trans_dict[filename]
            included_clusters = set(trans)

            for query_cluster in included_clusters:
                cluster_trues = np.array(trans) == query_cluster
                true_idx = np.where(cluster_trues)[0]

                splits = np.asarray(np.nonzero(np.asarray(true_idx[:-1]) -
                                           np.asarray(true_idx[1:]) != -1)) + 1

                splits = np.append(splits, len(true_idx))

                s = 0
                for t in splits:
                    e = t
                    start = true_idx[s:e].min()
                    end = true_idx[s:e].max()
                    s = e
                    fragments.append({'filename': filename,
                                         'start': start,
                                         'end': end,
                                         'cluster_id': query_cluster,
                                         })

        nodes_df = pd.DataFrame(fragments, columns=['filename','start','end','cluster_id'])

        return nodes_df


    @staticmethod
    def get_clusters(nodes_df):

        clusters_list = []

        for clus_id in nodes_df.cluster_id.unique():
            clusters_list.append(list(nodes_df.index[nodes_df.cluster_id == clus_id]))

        return clusters_list


    def get_nodes_clusters(self):

        self.trans_dict = self.get_transcriptions_dict()
        self.nodes_df = self.get_nodes(self.trans_dict)
        self.clusters_list = self.get_clusters(self.nodes_df)

        return self.nodes_df, self.clusters_list


    @staticmethod
    def compute_NMI_seq(pred, gold, verbose=False): 
        from sklearn.metrics import normalized_mutual_info_score
        
        if verbose: 
            print('random baseline: {:.3f}'.format(
            normalized_mutual_info_score(gold, np.random.randint(0,len(set(pred)),len(gold)))))

        return normalized_mutual_info_score(pred, gold)


    @staticmethod
    def compute_NMI(trans_dict, df):

        gold = []
        pred = []
        for fname in trans_dict.keys():
            gold.extend( list( df.label[df.folder == fname].values // 3 ))
            pred.extend(trans_dict[fname])

        assert(len(gold) == len(pred))

        nmi = BEER.compute_NMI_seq(pred, gold)

        return nmi
    


    def hmm_align(self, hmmname='final.mdl'):
        ''' given the features, align states using final hmm model '''
        # load the hmm model
        import torch

        hmmdir = os.path.join(self.outdir, hmmname)

        with open(hmmdir, 'rb') as fh:
            model = pickle.load(fh)

        # perform alignment
        ali_graphs = None
        aliz = dict()

        for uttid in self.uttids:

            ft = torch.from_numpy(self.feats_dict[uttid]).float()
            graph = None
            if ali_graphs is not None:
                graph = ali_graphs[uttid][0]
            aliz[uttid] = model.decode(ft, inference_graph=graph)

        # save as npz file
        np.savez(os.path.join(self.outdir, 'alis.npz'), **aliz)














































# feaname, exp_name, epoch = 10,    lr = 0.1, batch = 400
class BEERold:
    def __init__(self, uttids, feats, params):
        self.uttids = uttids
        self.feats = feats
        self.params = params
        self.root_dir = params['rootdir']


    def save_uttids(self):
        ### save filenames as 'uttids'
        self.data_unit_path = os.path.join(self.root_dir, 'data_unit', self.params['exp_name'], 'unit')
        os.makedirs(self.data_unit_path, exist_ok=True)
        with open(os.path.join(self.data_unit_path,'uttids'),'w') as f:
            for name in self.uttids:
                f.writelines(name + '\n')


    def save_npy_files(self):
        # save npy files
        self.features_unit_path = os.path.join(self.root_dir, 'features_unit',  self.params['exp_name'], 'unit')
        if os.path.exists(self.features_unit_path):
            shutil.rmtree(self.features_unit_path)
        os.makedirs(self.features_unit_path, exist_ok=True)
        os.makedirs(os.path.join(self.features_unit_path, self.params['feaname']), exist_ok=True)

        for i,name in enumerate(self.uttids):
            np.save(os.path.join(self.features_unit_path, self.params['feaname'], name), self.feats[i])


    def archive_to_zip(self):
        # arcihve to a single npz file
        counts = 0
        with ZipFile(self.features_unit_path + '/'+ self.params['feaname'] + '.npz', 'w') as f:
            for path in glob.glob(self.features_unit_path + '/' + self.params['feaname'] + '/*.npy'):
                arcname = os.path.basename(path).replace('.npy' , '')
                f.write(path, arcname=arcname)
                counts += 1


    def create_outdir(self):
        # create outdir
        pth = os.path.join(self.root_dir, 'exp', self.params['exp_name'], 'datasets')
        if os.path.exists(pth):
            shutil.rmtree(pth)
        os.makedirs(pth)

        pth = os.path.join(self.root_dir, 'exp', self.params['exp_name'], 'aud')
        if os.path.exists(pth):
            shutil.rmtree(pth)
        os.makedirs(pth)


    def get_commands(self):
        cmd_create_db = 'steps/create_dataset.sh {}  {}/{}.npz  exp/{}/datasets/unit.pkl'.format(
        self.data_unit_path, self.features_unit_path, self.params['feaname'], self.params['exp_name'])


        cmd_aud = 'steps/aud.sh conf/hmm.yml exp/{}/datasets/unit.pkl {} {} {} exp/{}/aud trans_unit.txt'.format(
                    self.params['exp_name'], self.params['epoch'], self.params['lr'], self.params['batch'], self.params['exp_name'])

        return cmd_create_db, cmd_aud


    def prepare(self):

        self.save_uttids()
        self.save_npy_files()
        self.archive_to_zip()
        self.create_outdir()
        cmd_create_db, cmd_aud = self.get_commands()

        print(cmd_create_db)
        print('')
        print(cmd_aud)


    def read_transcription(self):
        trans_file = os.path.join(self.root_dir, 'exp', self.params['exp_name'], 'aud', 'trans_unit.txt')

        with open(trans_file, 'r') as f:
            lines = f.readlines()

        return lines


    def read_to_dict(self, lines):
        # read them into a dict
        word_clusters = set()
        trans_dict = dict()
        for line in lines:
            filename = line.strip('\n').split(' ')[0]
            trans = line.strip('\n').split(' ')[1:]
            trans_dict[filename] = trans
            word_clusters |= set(trans)

        self.word_clusters = word_clusters

        return trans_dict


    def get_transcriptions(self):

        lines = self.read_transcription()
        self.trans_dict = self.read_to_dict(lines)

        return self.trans_dict


    def get_nodes(self, trans_dict):
        fragments = []

        for filename in self.uttids:

            trans = trans_dict[filename]
            included_clusters = set(trans)

            for query_cluster in included_clusters:
                cluster_trues = np.array(trans) == query_cluster
                true_idx = np.where(cluster_trues)[0]

                splits = np.asarray(np.nonzero(np.asarray(true_idx[:-1]) -
                                           np.asarray(true_idx[1:]) != -1)) + 1

                splits = np.append(splits, len(true_idx))

                s = 0
                for t in splits:
                    e = t
                    start = true_idx[s:e].min()
                    end = true_idx[s:e].max()
                    s = e
                    fragments.append({'filename': filename,
                                         'start': start,
                                         'end': end,
                                         'cluster_id': query_cluster,
                                         })

        nodes_df = pd.DataFrame(fragments, columns=['filename','start','end','cluster_id'])

        return nodes_df


    def get_clusters(self, nodes_df):

        clusters_list = []

        for clus_id in nodes_df.cluster_id.unique():
            clusters_list.append(list(nodes_df.index[nodes_df.cluster_id == clus_id]))

        return clusters_list


    def get_nodes_clusters(self):

        self.trans_dict = self.get_transcriptions()
        self.nodes_df = self.get_nodes(self.trans_dict)
        self.clusters_list = self.get_clusters(self.nodes_df)

        return self.nodes_df, self.clusters_list


    def compute_NMI(self, trans_dict, df):
        from sklearn.metrics import normalized_mutual_info_score

        gold = []
        pred = []
        for fname in trans_dict.keys():
            gold.extend( list( df.label[df.folder == fname].values // 3 ))
            pred.extend(trans_dict[fname])

        assert(len(gold) == len(pred))

        print('random baseline: {:.3f}'.format(normalized_mutual_info_score(gold, np.random.randint(0,len(set(pred)),len(gold)))))

        nmi = normalized_mutual_info_score(pred, gold)

        return nmi



