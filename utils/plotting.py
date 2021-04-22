import matplotlib.pyplot as plt
import numpy as np

label_dict = {'coverage': 'Coverage',
                'ned':'NED', 
                'n_clus': '# Clusters',
              'purity': 'Purity',
              'avg_purity': 'Average Purity', 
              'clus_purity':'Cluster Purity', 
              'grouping_P': 'Grouping Precision',
              'grouping_R': 'Grouping Recall',
                'grouping_F': 'Grouping F-score',
              'boundary_P': 'Boundary Precision',
              'boundary_R': 'Boundary Recall',
                'boundary_F': 'Boundary F-score',
              'clus_purity_inv': 'Inverse Cluster Purity'}


def plot_curve(ax, results, ax_dict, label, annot_freq=2):
    # results: list of score dicts
    data = np.zeros((len(results), 2))
    annots = []
    for i,score in enumerate(results):
        data[i,0] = score[ax_dict['x']]
        data[i,1] = score[ax_dict['y']]
        annots.append(score['dtw'])

    ax.plot(data[:,0], data[:,1], label=label)
    ax.set_xlabel(label_dict[ax_dict['x']])
    ax.set_ylabel(label_dict[ax_dict['y']])
    for i,ann in enumerate(annots):
        
        if (i%annot_freq) != 0: continue
        ax.annotate(str(ann), 
             xy=(results[i][ax_dict['x']], results[i][ax_dict['y']]),  
             xycoords='data')

def plot_grid_exp(rowcol_tpl, final, candidates, metrics, xaxis='coverage', annot_freq=2):    
    r,c = rowcol_tpl

    fig, axes = plt.subplots(nrows=r, ncols=c, figsize=(16,16), squeeze=False)

    if candidates == 'all': candidates = np.arange(len(final))
    for i,y in enumerate(metrics):
        ax = axes[i//c,i%c]
        for j,(key,value) in enumerate(final.items()):
            if j not in candidates: continue
            plot_curve(ax, value, {'x': xaxis, 'y': y}, str(j)+ '_' + key, annot_freq)
    ax.legend()
    

#    plt.legend()


