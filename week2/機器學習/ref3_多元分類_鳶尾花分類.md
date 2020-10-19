#
```

```

# Iris_scatter圖
```
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris    #導入資料集iris
  
#載入資料集  
iris = load_iris()  
print(iris.data)          #輸出資料集  
print(iris.target)         #輸出真實標籤  
#獲取花卉兩列資料集  
DD = iris.data  
X = [x[0] for x in DD]  
print(X)  
Y = [x[1] for x in DD]  
print(Y)  
  
#plt.scatter(X, Y, c=iris.target, marker='x')
plt.scatter(X[:50], Y[:50], color='red', marker='o', label='setosa') #前50個樣本
plt.scatter(X[50:100], Y[50:100], color='blue', marker='x', label='versicolor') #中間50個
plt.scatter(X[100:], Y[100:],color='green', marker='+', label='Virginica') #後50個樣本
plt.legend(loc=2) #左上角
plt.show()
```

# 決策樹分類鳶尾花資料
```

# 下載資料集
iris.data
```
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pydotplus
 
if __name__ == "__main__":
   
	iris_feature_E = "sepal lenght", "sepal width", "petal length", "petal width"
	iris_feature = "the length of sepal", "the width of sepal", "the length of petal", "the width of petal"
	iris_class = "Iris-setosa", "Iris-versicolor", "Iris-virginica"
	
	data = pd.read_csv("iris.data", header=None)
	iris_types = data[4].unique()
	for i, type in enumerate(iris_types):
		data.set_value(data[4] == type, 4, i)
	x, y = np.split(data.values, (4,), axis=1)
	x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)
	print(y_test)
 
	model = DecisionTreeClassifier(criterion='entropy', max_depth=6)
	model = model.fit(x_train, y_train)
	y_test_hat = model.predict(x_test)
	with open('iris.dot', 'w') as f:
		tree.export_graphviz(model, out_file=f)
	dot_data = tree.export_graphviz(model, out_file=None, feature_names=iris_feature_E, class_names=iris_class,
		filled=True, rounded=True, special_characters=True)
	graph = pydotplus.graph_from_dot_data(dot_data)
	graph.write_pdf('iris.pdf')
	f = open('iris.png', 'wb')
	f.write(graph.create_png())
	f.close()
 
	# 畫圖
	# 橫縱各採樣多少個值
	N, M = 50, 50
	# 第0列的範圍
	x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
	# 第1列的範圍
	x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
	t1 = np.linspace(x1_min, x1_max, N)
	t2 = np.linspace(x2_min, x2_max, M)
	# 生成網格採樣點
	x1, x2 = np.meshgrid(t1, t2)
    # # 無意義，只是為了湊另外兩個維度
    # # 打開該注釋前，確保注釋掉x = x[:, :2]
	x3 = np.ones(x1.size) * np.average(x[:, 2])
	x4 = np.ones(x1.size) * np.average(x[:, 3])
	
  
  # 測試點
	x_show = np.stack((x1.flat, x2.flat, x3, x4), axis=1)
	print("x_show_shape:\n", x_show.shape)
 
	cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
	cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
	
  # 預測值
	y_show_hat = model.predict(x_show)
	print(y_show_hat.shape)
	print(y_show_hat)
	
  # 使之與輸入的形狀相同
	y_show_hat = y_show_hat.reshape(x1.shape)
	print(y_show_hat)
	plt.figure(figsize=(15, 15), facecolor='w')
	
  # 預測值的顯示
	plt.pcolormesh(x1, x2, y_show_hat, cmap=cm_light)
	print(y_test)
	print(y_test.ravel())
	
  # 測試資料
	plt.scatter(x_test[:, 0], x_test[:, 1], c=np.squeeze(y_test), edgecolors='k', s=120, cmap=cm_dark, marker='*')
	# 全部資料
	plt.scatter(x[:, 0], x[:, 1], c=np.squeeze(y), edgecolors='k', s=40, cmap=cm_dark)
	plt.xlabel(iris_feature[0], fontsize=15)
	plt.ylabel(iris_feature[1], fontsize=15)
	plt.xlim(x1_min, x1_max)
	plt.ylim(x2_min, x2_max)
	plt.grid(True)
	plt.title('yuanwei flowers regressiong with DecisionTree', fontsize=17)
	plt.show()
 
	# 訓練集上的預測結果
	y_test = y_test.reshape(-1)
	print(y_test_hat)
	print(y_test)
	# True則預測正確，False則預測錯誤
	result = (y_test_hat == y_test)
	acc = np.mean(result)
	print('accuracy: %.2f%%' % (100 * acc))
 
    # 過擬合：錯誤率
	depth = np.arange(1, 15)
	err_list = []
	for d in depth:
		clf = DecisionTreeClassifier(criterion='entropy', max_depth=d)
		clf = clf.fit(x_train, y_train)
		# 測試資料
		y_test_hat = clf.predict(x_test)
		# True則預測正確，False則預測錯誤
		result = (y_test_hat == y_test)
		err = 1 - np.mean(result)
		err_list.append(err)
		print(d, 'error ratio: %.2f%%' % (100 * err))
	plt.figure(figsize=(15, 15), facecolor='w')
	plt.plot(depth, err_list, 'ro-', lw=2)
	plt.xlabel('DecisionTree Depth', fontsize=15)
	plt.ylabel('error ratio', fontsize=15)
	plt.title('DecisionTree Depth and Overfit', fontsize=17)
	plt.grid(True)
	plt.show()
```
```
https://blog.csdn.net/OliverkingLi/article/details/80596229
```
# xgboost
```
# 要先安裝xgboost ===>!pip install xgboost

# 主程式
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
	iris_feature_E = "sepal lenght", "sepal width", "petal length", "petal width"
	iris_feature = "the length of sepal", "the width of sepal", "the length of petal", "the width of petal"
	iris_class = "Iris-setosa", "Iris-versicolor", "Iris-virginica"
	
	data = pd.read_csv("iris.data", header=None)
	iris_types = data[4].unique()
	for i, type in enumerate(iris_types):
		data.set_value(data[4] == type, 4, i)
	x, y = np.split(data.values, (4,), axis=1)
 
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.7, random_state=1)
 
	data_train = xgb.DMatrix(x_train, label=y_train)
	data_test = xgb.DMatrix(x_test, label=y_test)
	watchlist = [(data_test, 'eval'), (data_train, 'train')]
	param = {'max_depth':3, 'eta':1, 'silent':1, 'objective':'multi:softmax', 'num_class':3}
 
	bst = xgb.train(param, data_train, num_boost_round=10, evals=watchlist)
	y_hat = bst.predict(data_test)
	result = y_test.reshape(1, -1) == y_hat
	print('the accuracy:\t', float(np.sum(result)) / len(y_hat))
```
# ensemble(Adaboost)
```
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import classification_report

# 載入sklearn自帶的iris（鳶尾花）資料集
iris = load_iris()

# 提取特徵資料和目標資料
X = iris.data
y = iris.target

# 將資料集以9:1的比例隨機分為訓練集和測試集，為了重現隨機分配設置隨機種子，即random_state參數
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=188)

# 產生實體分類器對象
clf = ensemble.AdaBoostClassifier()

# 分類器擬合訓練資料
clf.fit(X_train, y_train)

# 訓練完的分類器對測試資料進行預測
y_pred = clf.predict(X_test)

# classification_report函數用於顯示主要分類指標的文本報告
print(classification_report(y_test, y_pred, target_names=['setosa', 'versicolor', 'virginica']))
```
```
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        11
  versicolor       0.83      1.00      0.91         5
   virginica       1.00      0.93      0.96        14

    accuracy                           0.97        30
   macro avg       0.94      0.98      0.96        30
weighted avg       0.97      0.97      0.97        30

```
