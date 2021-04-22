import matplotlib.pyplot as plt
import time
from os.path import join
import os

import skopt
from skopt import gp_minimize, space, forest_minimize, dummy_minimize, gbrt_minimize
from skopt.plots import plot_convergence, plot_evaluations, plot_objective, plot_regret, partial_dependence
from functools import partial
from skopt import gp_minimize, forest_minimize, dummy_minimize
from skopt.callbacks import TimerCallback, CheckpointSaver, DeltaYStopper, VerboseCallback

from utils.pipeline_wrappers import try_run_exp, cv_experiment



SPACE_ZR = [
           skopt.space.Real(0.35, 0.7, name='T', prior='uniform'),
           skopt.space.Real(0.3, 0.7, name='medthr', prior='uniform'),
           skopt.space.Integer(1, 30, name='dx',prior='uniform'),
           skopt.space.Integer(1, 15, name='dy',prior='uniform'),
           skopt.space.Integer(3, 40, name='D',prior='uniform'),
           skopt.space.Integer(3, 40, name='R',prior='uniform'),
           skopt.space.Real(0.1, 30.0, name='castthr',prior='uniform'),
           skopt.space.Real(0.1, 30.0, name='trimthr',prior='uniform'),
        ]

SPACE_KNN = [ 
            skopt.space.Integer(2, 5, name='a', prior='uniform'),
            skopt.space.Integer(2,4, name='lmin', prior='uniform'),
            skopt.space.Integer(11,18,   name='lmax', prior='uniform'),
            skopt.space.Integer(20, 200, name='k', prior='uniform'),
            skopt.space.Integer(4, 12, name='dim_fix', prior='uniform'),
            skopt.space.Real(0.01, 0.09, name='top_delta', prior='uniform'),
           skopt.space.Real(1e-6, 0.4, name='r',prior='uniform', transform='identity'),
           skopt.space.Real(0.05, 0.8, name='s',prior='uniform', transform='identity'),
           skopt.space.Categorical(['','PCA40','PCAW40'], name='pca'),
           skopt.space.Categorical([True,False], name='norm', ),            
        ]

SPACE_SDTW = [ 
            skopt.space.Integer(4, 15, name='L', prior='uniform'),
            skopt.space.Integer(1, 10, name='extend_r', prior='uniform'),
        ]

SPACE_FUSE = [ 
            skopt.space.Real(0.1, 2, name='scale_param', prior='uniform'),
        ]

space_select = {'zr': SPACE_ZR, 'zr_cat': SPACE_ZR, 'knn': SPACE_KNN, 'sdtw': SPACE_SDTW, 'fuse': SPACE_FUSE}



class TunerSkopt():
    def __init__(self, seq_names, feats_dict, params):
        
        self.chkpoint_root = '/home/korhan/Dropbox/tez/checkpoints/' + params['disc_method']
        self.seq_names = seq_names
        self.feats_dict = feats_dict
        
        self.params = params
        
        self.tune_name = join(self.chkpoint_root, 
                              '{}_{}_{}_{}'.format(params['tune']['minimizer'], 
                                                params['CVset'], params['featype'],
                                                '_'.join(params['tune']['keys']) ))
        
        self.prepared = False
            

    def _set_callbacks(self, names=['saver', 'stopper', 'timer', 'verb']):
        os.makedirs(self.chkpoint_root, exist_ok=True)
                
        checkpoint_saver = CheckpointSaver(self.tune_name + '.pkl', compress=9, 
                                           store_objective=False)
        stopper = DeltaYStopper(self.params['tune']['stop_margin'], 
                                self.params['tune']['stopper_patience'])
        timer = TimerCallback()
        verb = VerboseCallback(n_total=1)

        callbacks = {'saver':checkpoint_saver, 'stopper': stopper, 
                     'timer': timer, 'verb': verb}
        
        self.callbacks = [callbacks[x] for x in names]

        
    def _space_setter(self, space_def=None):
        # selects the given keys from pre-defined space
        # space_def: list of strings, keys of which params to tune        
        
        if space_def is None:
            self.space_def = self.params['tune']['keys']
        
        SPACE_0 = space_select[self.params['disc_method']]        
        self.SPACE = [x for x in SPACE_0 if x.name in self.space_def]
        
        # default params
        self.x0 = [self.params['disc'][k] for k in self.space_def]
        
    
    def _load_checkpoint(self, tune_name=None):

        if tune_name is None:
            tune_name = self.tune_name
        
        if '.pkl' not in tune_name: tune_name += '.pkl'

        if os.path.isfile(tune_name):
            self.saved_res = skopt.load(tune_name)
            self.x0 = self.saved_res.x_iters
            self.y0 = self.saved_res.func_vals
            print('Checkpoint {} with score {:.2f} loaded !!!'.format(
                                                                tune_name,  
                                                                min(self.y0)) )

   

    def prepare(self, load_checkpoint=None, space_def=None):
        
        self._set_callbacks()
        
        self._space_setter(space_def)
        self.y0 = None
        
        if load_checkpoint is not None:
            if isinstance(load_checkpoint, str):
                self._load_checkpoint(load_checkpoint)
            else:
                self._load_checkpoint()
        
        
        minimizers = {'rf': forest_minimize, 'gp': gp_minimize, 'gbrt': gbrt_minimize}
        
        self.minimizer = minimizers[self.params['tune']['minimizer']]
        
        self.prepared = True

        
    def run(self, n_calls=20, random_state=42, **kwargs):
    
        if not self.prepared:
            self.prepare()

        @skopt.utils.use_named_args(self.SPACE)
        def objective(**space_params):

            self.params['disc'] = {**self.params['disc'], **space_params}
            print(' === Running for new space {} ==='.format(space_params))
#                                            [k,v for k,v in space_params.items()]) )

            tstart = time.time()

            if self.params['tune']['n_split'] > 1:
                scores = cv_experiment(self.seq_names, self.feats_dict, 
                                       self.params, nfold=self.params['tune']['n_split'])
            else:
                scores = try_run_exp(self.feats_dict, self.params, genname=True)

            telapsed = time.time() - tstart
            print(' === Experiment completed in {}m {:.1f}seconds ==='.format(
                                           int(telapsed//60), telapsed%60 ))

            if not isinstance(scores, dict): 
                return 1.
            else:
                return scores['ned']
        
        self.result = self.minimizer(objective, self.SPACE, n_calls=n_calls, 
                                x0=self.x0, y0=self.y0, 
                                random_state=random_state, 
                                callback=self.callbacks,
                                **kwargs)
        
        return self.result.x, self.result.fun
    
    
    def _plot(self, names=['scores', 'evals', 'partial']):
        
        if 'scores' in names:
            plt.plot(self.result.func_vals)
        if 'evals' in names:
            plot_evaluations(self.result)
        if 'partial' in names:
            plot_objective(self.result)
