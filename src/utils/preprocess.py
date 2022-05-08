import os, csv
import numpy as np
from numpy.core.numeric import identity
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
import networkx as nx
from sklearn.cluster import SpectralClustering
from sklearn import preprocessing
from sklearn import metrics
from graph_tool.all import *
# internal lib
import network as sn


def wilcoxon_test(x, label_list):
    # data = arr[:, :-1].reshape(-1, arr.shape[0])
    # label = arr[:, -1]
    # print(data.shape)
    # print(label.shape)

    p_values = []
    for i in range(len(x)):
        p = stats.wilcoxon(x[i], label_list).pvalue
        # rej_status, corrected_p = fdrcorrection(p, alpha=0.05)
        p_values.append(p)
        # p_values.append(corrected_p, i])
    rej_status, corrected_p = fdrcorrection(p_values, alpha=0.001)
    # print(rej_status)
    # rej_num = len(np.where(rej_status==False))
    # print('number of rejected features:', rej_num)
    rej_indices = np.where(corrected_p >= 0.001)
    
    keep_num = (corrected_p < 0.001).sum()
    print('number of preserved features:', keep_num)
    # print('corrected p values:', corrected_p)
    
    proc_x = np.delete(x, rej_indices, axis=0)
    print('feature number after wilcoxon:', len(proc_x))
    
    return proc_x


def construct_psn(path, dataset, sample_id):
    graph = sn.compute_adjacency_matrix(dataset, perform_wgcna=True, beta_grid=[2, 20, 2])

    graph_df = pd.DataFrame(graph.values)
    graph_df.to_csv(path+"new_adjmat.csv")
        
        
def weighted_degree(G, result_dict):
    degree_centrality_dict = dict()
    for n, v in G.degree(weight="weight"):
        degree_centrality_dict[n] = v

    # print('weighted degree:', degree_centrality_dict)
    print('weighted degree finish')

    for k, v in degree_centrality_dict.items():
        result_dict[k] = [v]


def current_flow_closeness_centrality(G, result_dict):
    current_flow_closeness_centrality_dict = nx.current_flow_closeness_centrality(G, weight="weight", solver='cg')
    # print('current flow closeness centrality:', current_flow_closeness_centrality_dict)
    print('current flow closeness centrality finish')

    for k, v in current_flow_closeness_centrality_dict.items():
        result_dict[k].append(v)


def current_flow_betweenness_centrality(G, result_dict):
    current_flow_betweenness_centrality_dict = nx.current_flow_betweenness_centrality(G, weight="weight", solver='cg')
    # print('current flow betweenness centrality', current_flow_betweenness_centrality_dict)
    print('current flow betweenness centrality finish')

    for k, v in current_flow_betweenness_centrality_dict.items():
        result_dict[k].append(v)


def eigen_vector(G, result_dict):
    eigenvector_centrality_dict = nx.eigenvector_centrality(G, weight="weight", max_iter=10000)
    # print('eigen vector:', eigenvector_centrality_dict)
    print('eigen vector finish')

    for k, v in eigenvector_centrality_dict.items():
        result_dict[k].append(v)


def katz_centrality(G, result_dict):
    katz_centrality_dict = nx.katz_centrality_numpy(G, weight="weight")
    # print('katz centrality:', katz_centrality_dict)
    print('katz centrality finish')

    for k, v in katz_centrality_dict.items():
        result_dict[k].append(v)


def load_centrality(G, result_dict):
    load_centrality_dict = nx.load_centrality(G, weight="weight")
    # print('load centrality:', load_centrality_dict)
    print('load centrality finish')

    for k, v in load_centrality_dict.items():
        result_dict[k].append(v)


def closeness_centrality(G, result_dict):
    closeness_centrality_dict = nx.closeness_centrality(G, distance="distance")
    # print('closeness centrality:', closeness_centrality_dict)
    print('closeness centrality finish')

    for k, v in closeness_centrality_dict.items():
        result_dict[k].append(v)


def pagerank_centrality(G, result_dict):
    pagerank_dict = nx.algorithms.link_analysis.pagerank_alg.pagerank(G, weight='weight', max_iter=1000)
    # print('page-rank centrality:', pagerank_dict)
    print('page-rank centrality finish')

    for k, v in pagerank_dict.items():
        result_dict[k].append(v)


def hits_centrality(G, result_dict):
    hits_hub_dict, hits_authority_dict = nx.algorithms.link_analysis.hits_alg.hits(G, max_iter=10000)
    # print('hits hub:', hits_hub_dict)
    # print('hits authority:', hits_authority_dict)
    print('hits finish')

    for k, v in hits_hub_dict.items():
        result_dict[k].append(v)

    for k, v in hits_authority_dict.items():
        result_dict[k].append(v)


def local_clustering_coefficient(G, result_dict):
    local_clustering_coefficient_dict = nx.clustering(G, weight="weight")
    # print('local clustering coefficient:', local_clustering_coefficient_dict)
    print('local clustering coefficient finish')

    for k, v in local_clustering_coefficient_dict.items():
        result_dict[k].append(v)


def iter_degree_centrality(G, result_dict):
    iter_degree_centrality_dict = dict()
    iter_G = G.copy()
    while iter_G:
        max_degree_v = -1
        for n, v in iter_G.degree(weight="weight"):
            if v > max_degree_v:
                max_degree_v = v
                max_degree_n = n
        iter_degree_centrality_dict[max_degree_n] = max_degree_v
        iter_G.remove_node(max_degree_n)

    # print('iterative degree centrality:', iter_degree_centrality_dict)
    print('iterative degree centrality finish')

    for k, v in iter_degree_centrality_dict.items():
        result_dict[k].append(v)


def iter_local_clustering_coefficient(G, result_dict):
    iter_local_clustering_coefficient_dict = dict()
    iter_G = G.copy()
    while iter_G:
        max_coef_v = -1
        for n, v in nx.clustering(iter_G, weight="weight").items():
            if v > max_coef_v:
                max_coef_v = v
                max_coef_n = n
        iter_local_clustering_coefficient_dict[max_coef_n] = max_coef_v
        iter_G.remove_node(max_coef_n)
    # print('iterative local clustering coefficient:', iter_local_clustering_coefficient_dict)
    print('iterative local clustering coefficient finish')

    for k, v in iter_local_clustering_coefficient_dict.items():
        result_dict[k].append(v)


def spectral_cluster(G, result_dict):
    adj_mat = nx.to_numpy_matrix(G)
    best_score = -1e9
    best_result = []
    ground_truth = np.zeros(92)  # 92?
    for i in range(92):
        ground_truth[i] = i

    for i in range(2, 15):
        sc = SpectralClustering(i, affinity='precomputed', n_init=100)
        sc.fit(adj_mat)
        score = metrics.silhouette_score(adj_mat, sc.labels_) # (b - a) / max(a, b)
        # print('score:', score)
        if best_score < score:
            best_score = score
            best_result = sc.labels_

    print('best_score:', best_score)
    print('best_result:', best_result)


    number_of_modules = 0
    for i in best_result:
        number_of_modules = max(number_of_modules, i)

    node_name_list = list(G.nodes())
    for i in range(len(node_name_list)):
        module = best_result[i]
        l = [0 for i in range(number_of_modules+1)]
        l[module]=1
        result_dict[node_name_list[i]]+=l
        
        
def stochastic_block_model(path, G, result_dict):
    graph = load_graph_from_csv(path+"new_adjmatrix.csv", directed=False, eprop_types=['float'],eprop_names=['weight'],string_vals = True)
    index = pd.read_csv(path+"new_adjmatrix_index.csv",sep=',', header=None, index_col=None)
    blockstate = minimize_blockmodel_dl(graph, deg_corr=True, state_args=dict(recs=[graph.ep.weight],rec_types=["real-exponential"]))
    mcmc_equilibrate(blockstate, wait=500,nbreaks=3,mcmc_args=dict(niter=10),verbose=True)

    blocks = blockstate.collect_vertex_marginals().get_2d_array(np.arange(blockstate.B)).T

    blocks_list = blocks.tolist()
    node_name_list = list(G.nodes())
    for i in range(len(node_name_list)):
        result_dict[node_name_list[i]]+=blocks_list[i]