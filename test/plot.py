from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pandas as pd


# 导入并清洗数据
data = pd.read_csv('RFM.csv')
data.user_id = data.user_id.astype('str')
print(data.info())
print(data.describe())
X = data.values[:,1:]

# 数据标准化(z_score)
Model = preprocessing.StandardScaler()
X = Model.fit_transform(X)

# 迭代，选择合适的K
ch_score = []
ss_score = []
inertia = []
for k in range(2,10):
    clf = KMeans(n_clusters=k,max_iter=1000)
    pred = clf.fit_predict(X)
    ch = metrics.calinski_harabaz_score(X,pred)
    ss = metrics.silhouette_score(X,pred)
    ch_score.append(ch)
    ss_score.append(ss)
    inertia.append(clf.inertia_)

# 做图对比
fig = plt.figure()
ax1 = fig.add_subplot(131)
plt.plot(list(range(4,12)),ch_score,label='ch',c='y')
plt.title('CH(calinski_harabaz_score)')
plt.legend()

ax2 = fig.add_subplot(132)
plt.plot(list(range(4,12)),ss_score,label='ss',c='b')
plt.title('轮廓系数')
plt.legend()

ax3 = fig.add_subplot(133)
plt.plot(list(range(4,12)),inertia,label='inertia',c='g')
plt.title('inertia')
plt.legend()
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.serif'] = ['SimHei']  # 设置正常显示中文
plt.show()
