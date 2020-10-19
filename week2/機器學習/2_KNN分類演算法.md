# 分類演算法
```
KNN演算法==「k nearest neighbor」 ===「k個最近的鄰居」

https://ithelp.ithome.com.tw/articles/10197110
```
```
K-Nearest Neighbor (KNN) Classification Algorithm
https://www.slideshare.net/palinanalytics/knearest-neighbor-knn-classification-algorithm
```
### sklearn.neighbors.KNeighborsClassifier
```

sklearn.neighbors.KNeighborsClassifier
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

class sklearn.neighbors.KNeighborsClassifier(
n_neighbors=5, *, 
weights='uniform', 
algorithm='auto', 
leaf_size=30, 
p=2,
metric='minkowski', 
metric_params=None, 
n_jobs=None, **kwargs)
```
### 範例說明
```
X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)

# 訓練
neigh.fit(X, y)

# 預測
print(neigh.predict([[1.1]]))
print(neigh.predict_proba([[0.9]]))
```
```
# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets.samples_generator import make_blobs
# 生成資料
centers = [[-2, 2], [2, 2], [0, 4]]
X, y = make_blobs(n_samples=60, centers=centers, random_state=0, cluster_std=0.60)

# 畫出數據
plt.figure(figsize=(16, 10))
c = np.array(centers)
plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap='cool');         # 畫出樣本
plt.scatter(c[:, 0], c[:, 1], s=100, marker='^', c='orange');   # 畫出中心點

from sklearn.neighbors import KNeighborsClassifier
# 先建立分類器(物件)
# 再使用分類器的fit()訓練
# 模型訓練
k = 5
clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(X, y);

# 進行預測
X_sample = [0, 2]
X_sample = np.array(X_sample).reshape(1, -1)
y_sample = clf.predict(X_sample);
neighbors = clf.kneighbors(X_sample, return_distance=False);

# 畫出示意圖
plt.figure(figsize=(16, 10))
plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap='cool')    # 樣本
plt.scatter(c[:, 0], c[:, 1], s=100, marker='^', c='k')   # 中心點
plt.scatter(X_sample[0][0], X_sample[0][1], marker="x", 
            s=100, cmap='cool')    # 待預測的點

for i in neighbors[0]:
    # 預測點與距離最近的 5 個樣本的連線
    plt.plot([X[i][0], X_sample[0][0]], [X[i][1], X_sample[0][1]], 
             'k--', linewidth=0.6);
```
