from scipy.io import mmread
import pandas as pd
'''
# 读取 .mtx 文件
mtx_file = "C:/Users/Administrator/Desktop/173822f5062e7efdfbc8a9cbeef1494a_b9ffde5fe7164d6514f0f9076577f89c_8/cbb3a7553ece7eaddeb8a56df781ccb0_efec6c51e58ef7a07c81dfa762c06d4f_8.mtx"
matrix = mmread(mtx_file).tocsc()

# 将矩阵转化为 DataFrame
df = pd.DataFrame(matrix.toarray())

# 保存为 CSV 文件
df.to_csv("C:/Users/Administrator/Desktop/173822f5062e7efdfbc8a9cbeef1494a_b9ffde5fe7164d6514f0f9076577f89c_8/output.csv", index=False)
'''

from scipy.sparse import coo_matrix
import pandas as pd

# 定义文件路径
file_path = "C:/Users/Administrator/Desktop/张红雨开题/173822f5062e7efdfbc8a9cbeef1494a_b9ffde5fe7164d6514f0f9076577f89c_8.txt"  # 替换为您的实际文件路径
output_csv_path = "C:/Users/Administrator/Desktop/张红雨开题/gene_expression_matrix.csv"  # 输出文件路径

# 读取文件内容
with open(file_path, 'r', encoding='latin1') as file:  # 指定 UTF-8 编码
    lines = file.readlines()

# 过滤注释行和空行
lines = [line.strip() for line in lines if not line.startswith('%') and line.strip()]

# 提取矩阵的维度信息
n_rows, n_cols, n_elements = map(int, lines[0].split())  # 第一行是矩阵的维度信息

# 解析非零值
data = []
for line in lines[1:]:
    row, col, value = map(int, line.split()[:3])  # 提取行号、列号和值
    data.append((row, col, value))

# 构建稀疏矩阵
rows, cols, values = zip(*data)
matrix = coo_matrix((values, (rows, cols)), shape=(n_rows, n_cols))

# 转换为 DataFrame
df = pd.DataFrame(matrix.toarray())

# 保存为 CSV 文件
df.to_csv(output_csv_path, index=False)
print(f"基因表达矩阵已成功保存为: {output_csv_path}")
