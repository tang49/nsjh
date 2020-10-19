# K-means演算法
```
K-means是機器學習中一個比較常用的演算法，屬於無監督學習演算法，其常被用於資料的聚類，

只需為它指定簇的數量即可自動將資料聚合到多類中，
相同簇中的資料相似度較高，不同簇中資料相似度較低。

K-MEANS演算法是輸入聚類個數k，以及包含 n個資料物件的資料庫，
輸出滿足方差最小標準k個聚類的一種演算法。

k-means 演算法接受輸入量 k ；
然後將n個資料物件劃分為 k個聚類以便使得所獲得的聚類滿足：
同一聚類中的物件相似度較高；而不同聚類中的物件相似度較小。

核心思想
通過迭代尋找k個類簇的一種劃分方案，使得用這k個類簇的均值來代表相應各類樣本時所得的總體誤差最小。
k個聚類具有以下特點：各聚類本身儘可能的緊湊，而各聚類之間儘可能的分開。
k-means演算法的基礎是最小誤差平方和準則


K-menas的優缺點：
優點：
原理簡單
速度快
對大資料集有比較好的伸縮性

缺點：
需要指定聚類 數量K
對異常值敏感
對初始值敏感

K-means的聚類過程

其聚類過程類似於梯度下降演算法，建立代價函式並通過迭代使得代價函式值越來越小

適當選擇c個類的初始中心；
在第k次迭代中，對任意一個樣本，求其到c箇中心的距離，將該樣本歸到距離最短的中心所在的類；
利用均值等方法更新該類的中心值；
對於所有的c個聚類中心，如果利用（2）（3）的迭代法更新後，值保持不變，則迭代結束，否則繼續迭代。

該演算法的最大優勢在於簡潔和快速。演算法的關鍵在於初始中心的選擇和距離公式。
```
```
K means
https://www.slideshare.net/ssuserf88631/k-means-42435149

K-means Clustering
https://www.slideshare.net/annafensel/kmeans-clustering-122651195
```

```

機器學習- K-means clustering in Python(附程式碼介紹)
https://medium.com/@a4793706/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-k-means-clustering-in-python-%E9%99%84%E7%A8%8B%E5%BC%8F%E7%A2%BC%E4%BB%8B%E7%B4%B9-55c19bcf2280

https://github.com/qwp8510/Machine-Learning-K-means-clustering/blob/master/K-means%20Clustering%20in%20Python.ipynb
```

# 使用scikit-learn 的KMeans
```
df = pd.DataFrame({
    'x': [12, 20, 28, 18, 29, 33, 24, 45,60, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72],
    'y': [39, 36, 30, 52, 54, 46, 55, 59, 35, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24]
})

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)
kmeans.fit(df)

labels = kmeans.predict(df)
centroids = kmeans.cluster_centers_

fig = plt.figure(figsize=(5, 5))

colors = map(lambda x: colmap[x+1], labels)

plt.scatter(df['x'], df['y'], color=list(colors), alpha=0.5, edgecolor='k')
for idx, centroid in enumerate(centroids):
    plt.scatter(*centroid, color=colmap[idx+1])
plt.xlim(0, 200)
plt.ylim(0, 200)
plt.show()
```

# 自己動手做
```
# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

df = pd.DataFrame({
    'x': [12, 20, 28, 18, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72],
    'y': [39, 36, 30, 52, 54, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24]
})

np.random.seed(200)
k = 3
centroids = {
    i+1:[np.random.randint(0,80),np.random.randint(0,80)]
    for i in range(k)
}

fig = plt.figure(figsize=(5,5))
plt.scatter(df['x'],df['y'],color='k')
colmap = {1:'r',2:'g',3:'b'}
for i in centroids.keys():
    plt.scatter(*centroids[i],color=colmap[i])
plt.xlim(0,80)
plt.ylim(0,80)
plt.show()

def assignment(df,centroids):
    for i in centroids.keys():
        # sqrt((x1 - x2)^2 - (y1 - y2)^2)
        df['distance_from_{}'.format(i)] = (
            np.sqrt(((df['x']-centroids[i][0])**2)
                    + (df['y']-centroids[i][1])**2)
        )
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    df['closest'] = df.loc[:,centroid_distance_cols].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x:int(x.lstrip('distance_from_')))
    df['color'] = df['closest'].map(lambda x:colmap[x])
    return df

df = assignment(df,centroids)
print(df.head())

fig = plt.figure(figsize=(5,5))
plt.scatter(df['x'],df['y'],color=df['color'],alpha=0.5,edgecolor='k')
plt.xlim(0,80)
plt.ylim(0,80)
plt.show()

import copy

old_centroids = copy.deepcopy(centroids)

def update(k):
    for i in centroids.keys():
        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
        centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
    return k

centroids = update(centroids)
print(centroids)
print(old_centroids)
fig = plt.figure(figsize=(5,5))
ax = plt.axes()
plt.scatter(df['x'],df['y'],color=df['color'],alpha=0.5,edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i],color=colmap[i])
plt.xlim(0,80)
plt.ylim(0,80)
for i in old_centroids.keys():
    old_x = old_centroids[i][0]
    old_y = old_centroids[i][1]
    dx = (centroids[i][0] - old_centroids[i][0]) * 0.75
    dy = (centroids[i][1] - old_centroids[i][1]) * 0.75
    ax.arrow(old_x,old_y,dx,dy,head_width=2,head_length=3,fc=colmap[i],ec=colmap[i])
plt.show()

df = assignment(df,centroids)

fig = plt.figure(figsize=(5,5))
plt.scatter(df['x'],df['y'],color=df['color'],alpha=0.5,edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i],color=colmap[i])
plt.xlim(0,80)
plt.ylim(0,80)
plt.show()

while True:
    closest_centroids = df['closest'].copy(deep=True)
    centroids = update(centroids)
    df = assignment(df,centroids)
    if closest_centroids.equals(df['closest']):
        break
        
fig = plt.figure(figsize=(5,5))
plt.scatter(df['x'],df['y'],color=df['color'],alpha=0.5,edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i],color=colmap[i])
plt.xlim(0,80)
plt.ylim(0,80)
plt.show()
```


#
```
Kmeans實現運動員位置聚集
籃球球員比賽的資料。        
資料集地址：KEEL-dataset – Basketball data set

該資料集主要包括5個特徵（Features），共96行資料。
特徵描述：共5個特徵，每分鐘助攻數、運動員身高、運動員出場時間、運動員年齡和每分鐘得分數。
```

```
【Python資料探勘課程】二.Kmeans聚類資料分析及Anaconda介紹
https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/566046/#outline__3
```
