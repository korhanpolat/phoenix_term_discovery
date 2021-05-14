import numpy as np
import igraph as ig
from numba import jit


def make_symmetric(edges_mat):

    edges_mat += edges_mat.T

    # return edges_mat
    return np.triu(edges_mat,1)


def edges2_vertices(edges_mat):


    edges_nz = edges_mat.nonzero()
    
    weights = edges_mat[edges_nz]
    edges = [(edges_nz[0][t], edges_nz[1][t]) for t in range(len(edges_nz[0]))]
        
    # iGraph expects vertex names to be 0,1,2,..., so we need a mapping
    vertices = set()
    for e in edges: vertices |= set(e)
    vertex_mapping = list(vertices)
#    vertex_mapping.index(7)
    # update edges according to new mapping
    edges = [(vertex_mapping.index(e[0]), vertex_mapping.index(e[1])) for e in edges]

    return edges, weights, vertex_mapping


def edges_to_iGraph(edges_mat):

    edges, weights, vertex_mapping = edges2_vertices(edges_mat)

    g = ig.Graph()

    g.add_vertices(vertex_mapping)  # add a list of unique vertices to the graph
    g.add_edges(edges)  # add the edges to the graph..
    g.es['weight'] = weights

    g = g.simplify(combine_edges=sum)

    return g, vertex_mapping


def memberships_to_cluster_list(memberships):
    memberships = np.asarray(memberships)

    clusters_list = []
    for c in np.unique(memberships):
        clusters_list.append(np.nonzero(memberships == c)[0].tolist())
    return clusters_list


def cluster_adj_mat(edges_mat, q_thr=0.8, method='fastgreedy'):
    # weights = edges_mat[edges_mat.nonzero()]  # edge weights

    g, vertex_mapping = edges_to_iGraph(edges_mat)
    print('*** vertex mapping done ***')

    if method=='fastgreedy':
        dend = g.community_fastgreedy()


        if q_thr == 0:
            optimal_count = dend.optimal_count
        else:
            # func. below is added from: ...envs/tez/lib/python3.6/site-packages/igraph/clustering.py
            optimal_count = dend.optimal_count_ratio(ratio=q_thr)

        clus = dend.as_clustering(optimal_count)

    elif method=='louvain':
        clus = g.community_multilevel()

    memberships = clus.membership

    modularity = clus.modularity

    clusters_list = memberships_to_cluster_list(memberships)

    # 
    for i,clus in enumerate(clusters_list):
        clusters_list[i] = [ vertex_mapping[c]  for c in clus]

    return clusters_list, memberships, modularity


def find_single_nodes_in_matrix(edges_mat):
    # find isolated nodes in the graph
    assert edges_mat.shape[0] == edges_mat.shape[1] # i.e. square
    nodes_to_delete = []
    for i in range(edges_mat.shape[0]):
        if sum(edges_mat[i,:]) == 0 and sum(edges_mat[:,i]) == 0:
            nodes_to_delete.append(i)
    return nodes_to_delete


def remove_single_nodes(edges_mat, nodes_to_delete):
    # delete nodes that dont have any connection
    edges_mat = np.delete(edges_mat, np.asarray(nodes_to_delete), axis=0 )
    edges_mat = np.delete(edges_mat, np.asarray(nodes_to_delete), axis=1 )

    return edges_mat

