import pandas as pd
import argparse
from sklearn.metrics import average_precision_score, roc_curve, auc, precision_recall_curve
import numpy as np

#参数output预测的基因对及其分数  label真实的基因对   TFs一个基因集合包括转录因子的名称  一个集合包含所有基因的名称
def evaluateEPR(output, label, TFs, Genes):
	label_set = set(label['Gene1']+'|'+label['Gene2'])
	output= output.iloc[:len(label_set)]      #预测的基因对及分数
	EPR = len(set(output['Gene1']+'|' +output['Gene2']) & label_set) / (len(label_set)**2/(len(TFs)*len(Genes)-len(TFs)))
	return EPR

def evaluateAUPRratio(output, label, TFs, Genes):
	label_set_aupr = set(label['Gene1']+label['Gene2'])
	preds,labels,randoms = [] ,[],[]
	res_d = {}
	l = []
	p= []
	for item in (output.to_dict('records')):
			res_d[item['Gene1']+item['Gene2']] = item['EdgeWeight']
	for item in (set(label['Gene1'])):
			for item2 in  set(label['Gene1'])| set(label['Gene2']):
				if item+item2 in label_set_aupr:
					l.append(1)
				else:
					l.append(0)
				if item+ item2 in res_d:
					p.append(res_d[item+item2])
				else:
					p.append(-1)
	return average_precision_score(l,p), average_precision_score(l,p)/np.mean(l)

def evaluateAU1(output, label):
	score = output.loc[:, ['EdgeWeight']].values
	label_dict = {}
	for row_index, row in label.iterrows():
		label_dict[row[0] + row[1]] = 1
	test_labels = []
	for row_index, row in output.iterrows():
		tmp = row[0]+str(row[1])
		if tmp in label_dict:
			test_labels.append(1)
		else:
			test_labels.append(0)
	test_labels = np.array(test_labels, dtype=bool).reshape([-1, 1])
	fpr, tpr, threshold = roc_curve(test_labels, score)
	auc_area = auc(fpr, tpr)
	# aucs.append(auc_area)
	precision, recall, _thresholds = precision_recall_curve(test_labels, score)
	aupr_area = auc(recall, precision)
	return auc_area, aupr_area

def evaluateAU(output, label):
    score = output.loc[:, ['EdgeWeight']].values
    label_dict = {}
    
    # 修改：排序 Gene1 和 Gene2 组合，确保顺序无关
    for row_index, row in label.iterrows():
        sorted_pair = tuple(sorted([row[0], row[1]]))  # 按照字母顺序排序基因对
        label_dict[sorted_pair] = 1
    
    test_labels = []
    for row_index, row in output.iterrows():
        # 修改：排序 Gene1 和 Gene2 组合，确保顺序无关
        tmp = tuple(sorted([row[0], row[1]]))  # 同样对 output 中的 Gene1 和 Gene2 排序
        if tmp in label_dict:
            test_labels.append(1)
        else:
            test_labels.append(0)
    
    test_labels = np.array(test_labels, dtype=bool).reshape([-1, 1])
    fpr, tpr, threshold = roc_curve(test_labels, score)
    auc_area = auc(fpr, tpr)
    
    precision, recall, _thresholds = precision_recall_curve(test_labels, score)
    aupr_area = auc(recall, precision)
    
    return auc_area, aupr_area
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'Evaluate for the inferred results of GARNet from scRNA-seq data.')
	parser.add_argument('-p','--pred_file', type = str,
						default="C:/Users/Administrator/Desktop/单细胞2/GARNet-main/output/Inferred_result_500_ChIP-seq_mESC.csv",
                        help='Path to inferred results file. Required. \n')
	parser.add_argument('-n','--network', type = str,
                        default = "C:/Users/Administrator/Desktop/单细胞2/GARNet-main原/Datasets/500_ChIP-seq_mESC/500_ChIP-seq_mESC-network.csv",
                        help='Path to network file to print network statistics. Optional. \n')
	args = parser.parse_args()

	groud_truth = args.network
	
	dataset  = groud_truth.split('/')[-2]
	output = pd.read_csv(args.pred_file, header = 0, sep=',')
	label = pd.read_csv(groud_truth, header = 0, sep = ',')
	
	output = output.groupby([output['Gene1'], output['Gene2']], as_index=False).mean()
	output = output[output['Gene1'] != output['Gene2']]
	auc, aupr = evaluateAU(output, label)

	output['EdgeWeight'] = abs(output['EdgeWeight'])
	output = output.sort_values('EdgeWeight',ascending=False)
	
	TFs = set(label['Gene1'])
	Genes = set(label['Gene1'])| set(label['Gene2'])
	output = output[output['Gene1'].apply(lambda x: x in TFs)]
	output = output[output['Gene2'].apply(lambda x: x in Genes)]
	epr = evaluateEPR(output, label, TFs, Genes)
	aupr_1, aupr_ratio = evaluateAUPRratio(output, label, TFs, Genes)
	print("========================Evaluation of Dataset: ", dataset, "========================")
	print("The AUC is:", '{:.4f}'.format(auc))
	print("The AUPR is:",'{:.4f}'.format(aupr))
	print("The EPR is:", '{:.4f}'.format(epr))
	print("The AUPR is:", '{:.4f}'.format(aupr_1))
	print("The AUPR ratio is:", '{:.4f}'.format(aupr_ratio))
