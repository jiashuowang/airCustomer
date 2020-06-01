#-*- coding:utf-8 -*-
'''
    data_kmeans.py
    ~~~~~~~~~~~~~~~~~~~~~~~
    K-Means 聚类算法：
    sklearn的cluster模块提供了KMeans函数构建K-Means聚类模型，其基本语法如下。
        sklearn.cluster.KMeans(n_clusters=8, init='k-means++',
            n_init=10, max_iter=300, tol=0.0001,precompute_distances='auto',
            verbose=0, random_state=None, copy_x=True, n_jobs=1,algorithm='auto')
'''

# 导入模块
import pandas as pd
import os
# 导入KMeans（K均值）算法模型
from sklearn.cluster import KMeans
# 保存模型
from sklearn.externals import joblib
# 评估指标——轮廓系数,前者为所有点的平均轮廓系数，后者返回每个点的轮廓系数
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn import metrics

# 步骤1：加载数据
# 设置excel文件的路径地址
excelPath = os.getcwd() + '\out' + os.sep + 'std_result.xls'
excelFile = pd.ExcelFile(excelPath)
data = excelFile.parse('Sheet1')
# data = excelFile.parse(excelFile.sheet_names[0])
print(data.shape)


# 步骤2：创建KMeans算法模型

# 设置数据分类的个数
k = 5

# for i in range(6,8):
#     clf = KMeans(n_clusters=i)
#     y_predict = clf.fit_predict(data)
#     s = silhouette_score(data, y_predict)
#     CH = metrics.calinski_harabaz_score(data, y_predict)
#     interia = clf.inertia_
#     print("k = " + str(i) + "===========")
#     print(s)
#     print(CH)
#     print(interia)
#     print("=============================")
# 创建KMeans算法模型对象并设置聚类的个数
kmeans = KMeans(n_clusters=k, random_state=0)
print('KMeans模型创建……')

# 模型训练
y = kmeans.fit(data)
print('KMeans模型训练……')
print("k = 5===========")
y_predict = kmeans.fit_predict(data)
s = silhouette_score(data, y_predict)
CH = metrics.calinski_harabaz_score(data, y_predict)
interia = kmeans.inertia_
print(s)
print(CH)
print(interia)
# 保存模型
# joblib.dump(kmeans, './model/doc_cluster.pkl')
# kmeans = joblib.load('./model/doc_cluster.pkl')
# result = kmeans.predict(data)
# print(result)

# # 统计每个类别的样本个数
# r1 = pd.Series(kmeans.labels_).value_counts()
# print('统计每个类别的样本数量：\n{0}'.format(r1))
#
# # 统计每个类别的聚类中心数据
# r2 = pd.DataFrame(kmeans.cluster_centers_)
# print('统计每个类别的聚类中心数据:\n{0}'.format(r2))
#
# # 将数据整合连接
# r = pd.concat([r1, r2], axis=1)
#
# # 重新设置列索引名称
# r.columns = ['聚类个数'] + list(data.columns)
# print('KMeans聚类分析统计数据:\n{0}'.format(r))
#
# # 将最终的数据写入到excel文件中
# r.to_excel('./out/K-Means_cluster_data_result.xls')
# print('KMeans聚类分析完毕，统计数据写入到excel文件中.')
#
# # 步骤3：将分类标签值对位绑定到原始样本数据
# r = pd.concat([data, pd.Series(kmeans.labels_, index=data.index)], axis=1)
# # r = pd.concat([data, pd.Series(kmeans.labels_)], axis=1)
#
# # 重新设置列索引名称
# r.columns = list(data.columns) + ['聚类类别']
# print('最终样本数据集:\n{0}'.format(r))
#
# # 写入到指定的excel文件中
# r.to_excel('./out/KMeans_cluster_details_data_result.xls')
# print('最终详细数据写入到指定的excel文件中.')