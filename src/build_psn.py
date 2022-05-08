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
parser.add_argument('-d', '--dataset', type=str, default='SEQC', help='SEQC or TARGET')

args = parser.parse_args()
inpath = args.inpath
dataset = args.dataset

if dataset == 'SEQC':
    array_dataset = pd.read_csv(inpath+"arraydata_clean.tsv", sep='\t', index_col = 0)
    clinical_dataset = pd.read_csv(inpath+"clinical_data.tsv", sep='\t', index_col = 0)
    X = array_dataset.values
    Y = clinical_dataset.iloc[:, 1].values
elif dataset == 'TARGET':
    array_dataset = pd.read_csv(inpath + 'DataMat.csv', index_col = 0)
    clinical_dataset = pd.read_csv(inpath + 'ClinicalData.csv', index_col = 0)
    # print(clinical_dataset.head())
    X = array_dataset.values
    Y = clinical_dataset.values.reshape((-1,))
    # print(Y)
else: 
    print('unrecognized dataset type!')
    exit()

print(X.shape)
print(Y.shape)

# outcome_dict = dict()
# row = 0

# for i in array_dataset.columns:  # columns are patient samples
# 	row += 1
# 	outcome_dict[i] = Y[row-1]

# print("outcome_dict:")
# print(outcome_dict)

l = list(Y)
print(l)

new_X = wilcoxon_test(X, l)
new_dataset = pd.DataFrame(X)

construct_psn(inpath, new_dataset, array_dataset.columns)