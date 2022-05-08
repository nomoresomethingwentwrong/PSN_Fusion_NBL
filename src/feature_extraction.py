import sys, os
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import csv
import random
# import igraph
# import graph_tool as gt
import networkx as nx
# import utils.network as sn
# import matplotlib.pylab as plt
from scipy import stats
from sklearn.cluster import SpectralClustering
from sklearn import preprocessing
from sklearn import metrics
from graph_tool.all import *
import argparse

sys.path.append('utils/')
from preprocess import *

parser = argparse.ArgumentParser(description='extract network features')
parser.add_argument('-i', '--inpath', type=str, default='../GSE49710_original/', help='input path')
parser.add_argument('-d', '--dataset', type=str, default='SEQC', help='dataset: SEQC or TARGET')
parser.add_argument('-f', '--fusion', action="store_true", default=False, help='add this flag to extract features for network-level fusion dataset')

args = parser.parse_args()
inpath = args.inpath
dataset = args.dataset
fusion = args.fusion

if dataset == 'SEQC':
    if not fusion:
        array_dataset = pd.read_csv(inpath+"arraydata_clean.tsv", sep='\t', index_col = 0)
        clinical_dataset = pd.read_csv(inpath+"clinical_data.tsv", sep='\t', index_col = 0)
    else:
        # there is no dataset for the fused dataset, but we can read the single omice dataset instead, since we only need the sample index and labels
        array_dataset = pd.read_csv("../GSE49710_original/arraydata_clean.tsv", sep='\t', index_col=0)
        clinical_dataset = pd.read_csv("../GSE49710_original/clinical_data.tsv", sep='\t', index_col=0)
    # X = array_dataset.values
    Y = clinical_dataset.iloc[:, 1].values
elif dataset == 'TARGET':
    if not fusion:
        array_dataset = pd.read_csv(inpath + 'DataMat.csv', index_col = 0)
        clinical_dataset = pd.read_csv(inpath + 'ClinicalData.csv', index_col = 0)
    else:
        array_dataset = pd.read_csv("../TARGET_Methylation/DataMat.csv", index_col = 0)
        clinical_dataset = pd.read_csv('../TARGET_Methylation/ClinicalData.csv', index_col = 0)
    # print(clinical_dataset.head())
    # X = array_dataset.values
    Y = clinical_dataset.values.reshape((-1,))
    # print(Y)
else: 
    print('unrecognized dataset type!')
    exit()

# print(X.shape)
print(Y.shape)

outcome_dict = dict()
row = 0

for i in array_dataset.columns:  # columns are patient samples
	row += 1
	outcome_dict[i] = Y[row-1]

print("outcome_dict:")
print(outcome_dict)

graph = pd.read_csv(inpath+'new_adjmat.csv', index_col=0).values.tolist()
print(len(graph), len(graph[0]))
# print('graph:', graph)

G = nx.Graph()
# # gtG = gt.Graph()
adjmat=[]

sample_id = array_dataset.columns

for i in range(len(sample_id)):
    for j in range(len(sample_id)):
        G.add_edge(sample_id[i], sample_id[j], weight=graph[i][j])
        adjmat.append([sample_id[i], sample_id[j],graph[i][j]])

print('number of nodes:', G.number_of_nodes())
# print(adjmat)
# print(G.degree(weight="weight"))


with open(inpath+"new_adjmatrix.csv", 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows(adjmat)


node_list = list(G.nodes)
with open(inpath+"new_adjmatrix_index.csv", "w") as csvfile:
    writer = csv.writer(csvfile, delimiter = ',')
    writer.writerow(node_list)

res_dict = dict()

# compute centrality features
weighted_degree(G, res_dict)
current_flow_closeness_centrality(G, res_dict)
current_flow_betweenness_centrality(G, res_dict)
eigen_vector(G, res_dict)
katz_centrality(G, res_dict)
hits_centrality(G, res_dict)
iter_degree_centrality(G, res_dict)
load_centrality(G, res_dict)
closeness_centrality(G, res_dict)
pagerank_centrality(G, res_dict)
local_clustering_coefficient(G, res_dict)
iter_local_clustering_coefficient(G, res_dict)

# compute modularity features
spectral_cluster(G, res_dict)
stochastic_block_model(inpath, G, res_dict)

with open(inpath + 'new_result.csv', 'w', newline='') as csvfile:
	writer = csv.writer(csvfile, delimiter=',')
	for k,v in res_dict.items():
		writer.writerow(v)
  
result_csv = pd.read_csv(inpath + 'new_result.csv', header = None)
# result_csv.head()

result_np = result_csv.to_numpy()
print('shape of features matrix:', result_np.shape)


# result_np = np.array([v for k,v in result_dict.items()])
result_header_np = np.array([k for k, v in res_dict.items()])
# print(result_np.shape)
print(result_header_np.shape)


result_scaled = preprocessing.scale(result_np)
result_scaled_list = result_scaled.tolist()

# append Y
node_name_list = list(G.nodes())
for i in range(len(node_name_list)):
	result_scaled_list[i].append(outcome_dict[node_name_list[i]])
 
with open(inpath+'new_scaled_result.csv', 'w',  newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows(result_scaled_list)