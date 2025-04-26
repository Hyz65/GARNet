from __future__ import division
from __future__ import print_function
import numpy as np
import time
import tensorflow as tf
from util.utils import *
from garnet.models import GARNet,Discriminator
from scipy.stats import nbinom
import gc
print(tf.__version__)
tf.compat.v1.enable_eager_execution()
tf.config.run_functions_eagerly(True)
print("Eager execution:", tf.executing_eagerly())

import argparse
parser = argparse.ArgumentParser(description='Configuration for GARNet model training.')
parser.add_argument('--model', default='GARNet', type=str, help='Model string.')
parser.add_argument('--input_path', default='./Datasets/500_ChIP-seq_mESC/', type=str, help='Input data path.')
parser.add_argument('--output_path', default='./output/', type=str, help='Output data path.')
parser.add_argument('--learning_rate', default=0.0001, type=float, help='Initial learning rate.')
parser.add_argument('--discriminator_learning_rate', default=0.0001, type=float, help='Initial learning rate for the discriminator.')
parser.add_argument('--cv', default=3, type=int, help='Folds for cross validation.')
parser.add_argument('--epochs', default=100, type=int, help='Number of epochs to train.')
parser.add_argument('--hidden1', default=64, type=int, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', default=64, type=int, help='Number of units in hidden layer 2.')
parser.add_argument('--dropout', default=0.3, type=float, help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight for L2 loss on embedding matrix.')
parser.add_argument('--early_stopping', default=10, type=int, help='Tolerance for early stopping (# of epochs).')
parser.add_argument('--max_degree', default=3, type=int, help='Maximum Chebyshev polynomial degree.')
parser.add_argument('--ratio', default=1, type=int, help='Ratio of negative samples to positive samples.')
parser.add_argument('--dim', default=64, type=int, help='The size of latent factor vector.')
parser.add_argument('--latent_factor_num', default=64, type=int, help='Number of latent factors.')
args = parser.parse_args()

import tensorflow as tf
import numpy as np
import time
import gc
import pandas as pd

def train(args, adj, features, train_arr, test_arr, labels, AM, gene_names, TF, result_path):
    # 加载数据
    adj, size_gene, logits_train, logits_test, train_mask, test_mask, labels = load_data(
        adj, train_arr, test_arr, labels, AM)

    input_dim = features.shape[1]

    # 初始化模型
    if args.model == 'GARNet':
        model = GARNet(
            input_dim=input_dim,
            size_gene=size_gene,
            latent_factor_num=args.latent_factor_num,
            hidden1=args.hidden1,
            hidden2=args.hidden2,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            discriminator_learning_rate=args.discriminator_learning_rate,
            adj=adj
        )
    else:
        raise ValueError(f"Invalid argument for model: {FLAGS.model}")

    # 开始训练
    for epoch in range(args.epochs):
        gc.collect()
        start_time = time.time()

        # 生成真实分布
        real_distribution = tf.random.normal([adj.shape[0], args.hidden2])
        
        # 训练生成器和鉴别器
        #gen_loss, disc_loss,loss,acc = model.train_step1(features, logits_train, real_distribution)
        gen_loss, disc_loss = model.train_step2(features, logits_train, real_distribution)

        # 输出本轮结果
        #print(f"Epoch {epoch + 1}/{args.epochs} | Gen Loss: {gen_loss}, Disc Loss: {disc_loss},Loss:{loss},Time: {time.time() - start_time:.5f}s")
        print(f"Epoch {epoch + 1}/{args.epochs} | Gen Loss: {gen_loss}, Disc Loss: {disc_loss},Time: {time.time() - start_time:.5f}s")

    print("Training Finished!")

    # 测试模型
    print("Evaluating model on test set...")
    #gen_loss, disc_loss,loss,acc = model.train_step1(features, logits_test, real_distribution)
    gen_loss, disc_loss = model.train_step2(features, logits_test, real_distribution)
    #print(f"Test Results: Gen Loss: {gen_loss}, Disc Loss: {disc_loss},Loss:{loss}")
    print(f"Test Results: Gen Loss: {gen_loss}, Disc Loss: {disc_loss}")


    # 保存结果
    print("Saving results...")
    predictions = model.predict(features)
    predictions = predictions.numpy().reshape((size_gene, size_gene))

    logits_train = logits_train.reshape(predictions.shape)
    TF_mask = np.zeros(predictions.shape)
    for i, item in enumerate(gene_names):
        for j in range(len(gene_names)):
            if i == j or (logits_train[i, j] == 1):
                continue
            if item in TF:
                TF_mask[i, j] = 1

    geneNames = np.array(gene_names)
    idx_rec, idx_send = np.where(TF_mask)
    results = pd.DataFrame({'Gene1': geneNames[idx_rec], 'Gene2': geneNames[idx_send], 'EdgeWeight': predictions[idx_rec, idx_send]})
    results = results.sort_values(by=['EdgeWeight'], axis=0, ascending=False)

    results.to_csv(result_path, index=False)
    print(f"Results saved to {result_path}")

    return results
