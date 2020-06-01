
import pandas as pd
import os
import numpy as np  # 数据结构
import sklearn.cluster as skc  # 密度聚类
from sklearn import metrics   # 评估模型
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


excelPath = os.getcwd() + '/out' + os.sep + 'std_result.xls'
excelFile = pd.ExcelFile(excelPath)
data = excelFile.parse('Sheet1')

db = skc.DBSCAN(eps=0.615, min_samples=9).fit(data)
label_pred = db.labels_
print(label_pred)
n_clusters_ = len(set(label_pred)) - (1 if -1 in label_pred else 0)
print("n_clusters = " + str(n_clusters_) + "==========")
print(metrics.silhouette_score(data, label_pred))

# PCA降维
pca = PCA(n_components=2)
data = pd.DataFrame(pca.fit_transform(data))

# 绘制k-means结果
x0 = data[label_pred == 0]
x1 = data[label_pred == 1]
x2 = data[label_pred == 2]
x3 = data[label_pred == 3]
x4 = data[label_pred == 4]
plt.scatter(x0.iloc[:, 0], x0.iloc[:, 1], c="red", marker='o', label='label0')
plt.scatter(x1.iloc[:, 0], x1.iloc[:, 1], c="green", marker='*', label='label1')
plt.scatter(x2.iloc[:, 0], x2.iloc[:, 1], c="blue", marker='+', label='label2')
plt.scatter(x3.iloc[:, 0], x3.iloc[:, 1], c="yellow", marker='+', label='label3')
plt.scatter(x4.iloc[:, 0], x4.iloc[:, 1], c="grey", marker='+', label='label4')
plt.legend(loc=2)
plt.show()

# data['cluster_db'] = label_pred  # 在数据集最后一列加上经过DBSCAN聚类后的结果
# data.to_excel('./out/hhh.xls', index=False)
# # print(data['cluster_db'])
# # data.sort_values('cluster_db')
# # # plt.show(pd.plotting.scatter_matrix(data, c=data.cluster_db, figsize=(10,10),marker='1',alpha=.8,s=100))
# # plt.show(pd.plotting.scatter_matrix(data, c=data.cluster_db.tolist(), figsize=(10,10), s=100))
# plt.show(pd.plotting.scatter_matrix(data, alpha=0.7, figsize=(14,8), diagonal='kde'))

