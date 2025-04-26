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
parser.add_argument('--learning_rate', default=0.0001, type=float, help='Initial learning rate.')
parser.add_argument('--discriminator_learning_rate', default=0.0001, type=float, help='Initial learning rate for the discriminator.')
parser.add_argument('--cv', default=3, type=int, help='Folds for cross validation.')
parser.add_argument('--epochs', default=500, type=int, help='Number of epochs to train.')
parser.add_argument('--hidden1', default=300, type=int, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', default=500, type=int, help='Number of units in hidden layer 2.')
parser.add_argument('--dropout', default=0.4, type=float, help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight for L2 loss on embedding matrix.')
parser.add_argument('--early_stopping', default=10, type=int, help='Tolerance for early stopping (# of epochs).')
parser.add_argument('--max_degree', default=1, type=int, help='Maximum Chebyshev polynomial degree.')  #3
parser.add_argument('--ratio', default=1, type=int, help='Ratio of negative samples to positive samples.')
parser.add_argument('--dim', default=500, type=int, help='The size of latent factor vector.')
parser.add_argument('--latent_factor_num', default=500, type=int, help='Number of latent factors.')

args = parser.parse_args()


def computCorr(data, t = 0.0):

    genes = data.columns
    corr = data.corr(method = "spearman")

    adj = np.array(abs(corr.values))
    return adj

def prepareData(FLAGS, data_path, reverse_flags = 0):
    # Reading data from disk
    #label_file = pd.read_csv(label_path, header=0, sep = ',')
    data = pd.read_csv(data_path, header=0, index_col = 0).T                 ###transpose for six datasets of BEELINE
    
    print("Read data completed! Normalize data now!")
    data = data.transform(lambda x: np.log(x + 1))
    print("Data normalized and logged!")
    var_names = list(data.columns)
    
    print("Start to compute correlations between genes!")
    adj = computCorr(data)
    node_feat = data.T.values
    return  adj, var_names, node_feat   #边标签、基因数据之间的相关性矩阵、邻接矩阵（Adjacency Matrix）、基因的名称列表、节点特征矩阵


# Preparing data for training
input_path ="C:/Users/Administrator/Desktop/单细胞2/GARNet1-main/Datasets/500_ChIP-seq_mESC/"
output_path = "C:/Users/Administrator/Desktop/单细胞2/GARNet1-main/output/"
dataset = input_path.split('/')[-2]
data_file = input_path  + dataset +'-ExpressionData.csv'

reverse_flags = 0   ###whether label file exists reverse regulations, 0 for DeepSem data, 1 for CNNC data

adj, gene_names, node_feat = prepareData(FLAGS, data_file, reverse_flags)

reorder = np.arange(len(node_feat))  # Assuming `node_feat` is the data
np.random.shuffle(reorder)

# Enable eager execution for TensorFlow
tf.config.run_functions_eagerly(True)
print("Eager execution enabled:", tf.executing_eagerly())

T = 1  # Number of training rounds
cv_num = 1  # We don't need k-fold for a simple 70% - 30% split
start = time.time()

for t in range(T):
    print(f"T: {t}")
    
    # Split the data into 70% training and 30% testing
    split_index = int(len(reorder) * 0.7)  # 70% training data
    train_arr = reorder[:split_index]
    test_arr = reorder[split_index:]

    # Call the training function
    result_path_cv = output_path + '/pred_result_' + dataset + '_CV' + str(cv_num) + '.csv'
    pred_matrix = train(args, adj, node_feat, train_arr, test_arr, gene_names, result_path_cv)

    # Saving the results
    pred_matrix['EdgeWeight'] = abs(pred_matrix['EdgeWeight'])
    result_path = output_path + '/Inferred_result_' + dataset + '.csv'
    pred_matrix.to_csv(result_path, header=True, index=False)

print("Predict complete!")
print("RunTimes is:", "{:.5f}".format(time.time() - start))



