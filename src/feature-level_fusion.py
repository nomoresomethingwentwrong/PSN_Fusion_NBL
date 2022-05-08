import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='feature level fusion')
parser.add_argument('-d', '--dataset', type=str, default='SEQC', help='dataset type: SEQC or TARGET')

args = parser.parse_args()
dataset = args.dataset

if dataset == 'SEQC':
    file1 = '../GSE49710_original/new_scaled_result.csv'
    file2 = '../GSE62564_original/new_scaled_result.csv'
    file = '../GSE49710+62564/new_scaled_result_FeaFusion.csv'
elif dataset == 'TARGET':
    file1 = '../TARGET_Methylation/new_scaled_result.csv'
    file2 = '../TARGET_RNAseq/HTSeq_counts/new_scaled_result.csv'
    file = '../TARGET_Fusion/new_scaled_result_FeaFusion.csv'
else:
    print('unrecognized dataset type!')
    exit()
    
data1 = pd.read_csv(file1, header=None)
cen1 = data1.iloc[:, :13]
mod1 = data1.iloc[:, 13:]

data2 = pd.read_csv(file2, header=None)
cen2 = data2.iloc[:, :13]
mod2 = data2.iloc[:, 13:]

cen_fea1 = cen1.values
cen_fea2 = cen2.values
mod_fea1 = mod1.values
mod_fea2 = mod2.values

cen_fea = np.mean([cen_fea1, cen_fea2], axis = 0)
mod_fea = np.concatenate((mod_fea1, mod_fea2), axis = 1)
fea = np.concatenate((cen_fea, mod_fea), axis = 1)
df = pd.DataFrame(fea, dtype = 'float64')
df.to_csv(file, index=False, header=False)