from hyperopt import fmin, tpe, space_eval, hp, Trials, STATUS_OK, STATUS_FAIL
from hyperopt.pyll import stochastic, scope
from os.path import join
import os
import pandas as pd
import numpy as np


@scope.define
def round_n(x, n=3):
    return np.round(x, n)


def monitor_callback(params, scores, name=''):

	tmp = {'NED':scores['ned'],
			'Coverage': scores['coverageNS'],
			'scores': scores}

	tmp = {**params['disc'], **params['clustering'], **tmp}

	outfile = join(params['exp_root'], 'results', name + '_expresults.csv')
	if os.path.exists(outfile):
	    pd.DataFrame([tmp]).to_csv(outfile, mode='a', header=False)
	else: 
	    pd.DataFrame([tmp]).to_csv(outfile, mode='w', header=True)




def save_csv(params, tmp, name):
    outfile = join(params['exp_root'], 'results', name + '_expresults.csv')
    if os.path.exists(outfile):
        pd.DataFrame([tmp]).to_csv(outfile, mode='a', header=False)
    else: 
        pd.DataFrame([tmp]).to_csv(outfile, mode='w', header=True)



