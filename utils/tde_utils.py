import os
import sys

tde_root = '/home/korhan/Desktop/tdev2-master/'
sys.path.append(tde_root)

import pkg_resources 
from tdev2.readers.gold_reader import *
from tdev2.readers.disc_reader import *
from tdev2.measures.ned import *
from tdev2.measures.boundary import *
from tdev2.measures.grouping import *
from tdev2.measures.coverage import *
from tdev2.measures.token_type import *



def zr2tde(nodesfile, dedupsfile, outfile):
    # Decode nodes file
    nodes_ = dict()
    with open(nodesfile) as nodes:
        for n, node in enumerate(nodes, start=1):
            try:
                wavfile, start, end  = node.split()[:3]
                nodes_[n] = [wavfile, float(start)/1.0, float(end)/1.0] 
            except:
                raise 

    # decode dedups file
    dedups_ = list()
    with open(dedupsfile) as dedups:
        for dedup in dedups:
            try:
                dedups_.append([int(n) for n in dedup.split() ])  
            except:
                raise

    # creating the output class used by eval
    t_ = ''
    for n, class_ in enumerate(dedups_, start=1):
        t_ += 'Class {}\n'.format(n)
        for element in class_:
            file_, start_, end_ = nodes_[element]
            t_ += '{} {:.2f} {:.2f}\n'.format(file_, start_, end_)
        t_ += '\n'

    # stdout or save to file file
    if outfile is None:
        print(t_)
    else:
        with open(outfile, 'w') as output:
            output.write(t_) 


def zrexp2tde(exp_path):

    nodesfile = os.path.join(exp_path, 'results','master_graph.nodes')
    dedupsfile = os.path.join(exp_path, 'results','master_graph.dedups')
    outfile = os.path.join(exp_path, 'results','master_graph.class')
    zr2tde(nodesfile, dedupsfile, outfile)
    
    return outfile


def prf2dict(dct, measurename, obj):
    dct[measurename + '_P'] = obj.precision
    dct[measurename + '_R'] = obj.recall
    dct[measurename + '_F'] = obj.fscore
    
    return dct


def compute_scores(gold, disc, measures=[]):
    scores = dict()
    
    # Launch evaluation of each metric
    if len(measures) == 0 or "boundary" in measures:
        print('Computing Boundary...')
        boundary = Boundary(gold, disc)
        boundary.compute_boundary()
        scores = prf2dict(scores, 'boundary', boundary)
        
    if len(measures) == 0 or "grouping" in measures:
        print('Computing Grouping...')
        grouping = Grouping(disc)
        grouping.compute_grouping()
        scores = prf2dict(scores, 'grouping', grouping)    
        
    if len(measures) == 0 or "token/type" in measures:
        print('Computing Token and Type...')
        token_type = TokenType(gold, disc)
        token_type.compute_token_type()
        scores['token_P'],scores['token_R'],scores['token_F'] = token_type.precision[0], token_type.recall[0], token_type.fscore[0]
        scores['type_P'],scores['type_R'],scores['type_F'] = token_type.precision[1], token_type.recall[1], token_type.fscore[1]        
        
    if len(measures) == 0 or "coverage" in measures:
        print('Computing Coverage...')
        coverage = Coverage(gold, disc)
        coverage.compute_coverage()
        scores['coverage'] = coverage.coverage
        
    if len(measures) == 0 or "ned" in measures:
        print('Computing NED...')
        ned = Ned(disc)
        ned.compute_ned()
        scores['ned'] = ned.ned
    
    return scores