import tensorflow as tf
import numpy as np
import pandas as pd
from train import *
from util.utils import div_list
import time
import pdb
tf.config.run_functions_eagerly(True)
print("Eager execution enabled:", tf.executing_eagerly())

# Set random seed
seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)
import argparse


parser = argparse.ArgumentParser(description='Configuration for GARNet model training.')
parser.add_argument('--model', default='GARNet', type=str, help='Model string.')
parser.add_argument('--input_path', default='./Datasets/500_ChIP-seq_mESC/', type=str, help='Input data path.')
parser.add_argument('--output_path', default='./output/', type=str, help='Output data path.')
parser.add_argument('--learning_rate', default=0.00001, type=float, help='Initial learning rate.')
parser.add_argument('--discriminator_learning_rate', default=0.00001, type=float, help='Initial learning rate for the discriminator.')
parser.add_argument('--cv', default=3, type=int, help='Folds for cross validation.')
parser.add_argument('--epochs', default=100, type=int, help='Number of epochs to train.')
parser.add_argument('--hidden1', default=64, type=int, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', default=64, type=int, help='Number of units in hidden layer 2.')
parser.add_argument('--dropout', default=0.5, type=float, help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight for L2 loss on embedding matrix.')
parser.add_argument('--early_stopping', default=5, type=int, help='Tolerance for early stopping (# of epochs).')
parser.add_argument('--max_degree', default=2, type=int, help='Maximum Chebyshev polynomial degree.')  #3
parser.add_argument('--ratio', default=1, type=int, help='Ratio of negative samples to positive samples.')
parser.add_argument('--dim', default=64, type=int, help='The size of latent factor vector.')
parser.add_argument('--latent_factor_num', default=64, type=int, help='Number of latent factors.')

args = parser.parse_args()


def computCorr(data, t = 0.0):

    genes = data.columns
    corr = data.corr(method = "spearman")

    adj = np.array(abs(corr.values))
    return adj

def prepareData(FLAGS, data_path, label_path, reverse_flags = 0):
    # Reading data from disk
    label_file = pd.read_csv(label_path, header=0, sep = ',')
    data = pd.read_csv(data_path, header=0, index_col = 0).T                 ###transpose for six datasets of BEELINE
    
    print("Read data completed! Normalize data now!")
    data = data.transform(lambda x: np.log(x + 1))
    print("Data normalized and logged!")

    TF = set(label_file['Gene1'])
    # Adjacency matrix transformation
    labels = []
    if reverse_flags == 0:
        var_names = list(data.columns)
        num_genes = len(var_names)
        AM = np.zeros([num_genes, num_genes])
        for row_index, row in label_file.iterrows():
            AM[var_names.index(row[0]), var_names.index(row[1])] = 1
            label_triplet = []
            label_triplet.append(var_names.index(row[0]))
            label_triplet.append(var_names.index(row[1]))
            label_triplet.append(1)
            labels.append(label_triplet)

    labels = np.array(labels)
    print("Start to compute correlations between genes!")
    adj = computCorr(data)
    node_feat = data.T.values
    return labels, adj, AM, var_names, TF, node_feat   #边标签、基因数据之间的相关性矩阵、邻接矩阵（Adjacency Matrix）、基因的名称列表、节点特征矩阵


# Preparing data for training
input_path ="C:/Users/Administrator/Desktop/Datasets/500_ChIP-seq_mESC/"
output_path = "C:/Users/Administrator/Desktop/output/"
dataset = input_path.split('/')[-2]
data_file = input_path  + dataset +'-ExpressionData.csv'
#data_file = input_path +'ExpressionData.csv'
label_file = input_path + dataset + '-network.csv'



reverse_flags = 0   ###whether label file exists reverse regulations, 0 for DeepSem data, 1 for CNNC data

labels, adj, AM, gene_names,TF, node_feat = prepareData(FLAGS, data_file, label_file, reverse_flags)
reorder = np.arange(labels.shape[0])
np.random.shuffle(reorder)

tf.config.run_functions_eagerly(True)
print("Eager execution enabled:", tf.executing_eagerly())

T = 1  # Number of training rounds
cv_num = args.cv  # k-flod Cross-validation (CV)
start = time.time()
for t in range(T):
    order = div_list(reorder.tolist(), cv_num)
    pred_results = []
    for i in range(cv_num):
        print("T:", '%01d' % (t))
        print("cross_validation:", '%01d' % (i))
        result_path_cv = output_path + '/pred_result_' + dataset + '_CV' + str(cv_num) + '_' + str(i) + '.csv'
        test_arr = order[i]  #测试样本
        arr = list(set(reorder).difference(set(test_arr)))
        np.random.shuffle(arr)
        train_arr = arr  #训练样本
        pred_matrix = train(args, adj, node_feat, train_arr, test_arr, labels,AM, gene_names, TF, result_path_cv)
        pred_results.append(pred_matrix)
        
    output = pred_results[0]
    for i in range(1, cv_num):
        output = pd.concat([output, pred_results[i]])
    output['EdgeWeight'] = abs(output['EdgeWeight'])

    result_path = output_path + '/Inferred_result_' + dataset + '.csv'
    #result_path="C:/Users/Administrator/Desktop/PBCM/TNBCs/1.csv"
    output.to_csv(result_path, header=True, index=False)
        
print("Predict complete!")
print("RunTimes is:", "{:.5f}".format(time.time() - start))


