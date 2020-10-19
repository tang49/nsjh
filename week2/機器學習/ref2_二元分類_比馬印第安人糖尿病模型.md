# 比馬印第安人糖尿病 (Pima Indians Diabetes)
```
該資料根據醫療記錄預測比馬印第安人 5 年內糖尿病的發病情況。

它是一個二元分類問題
一共有 768 個樣例。

該資料集包含了 8 個特徵和 1 個類變數：

懷孕次數
2 小時的血漿葡萄糖濃度。
舒張壓
三頭肌皮膚褶層厚度
2 小時血清胰島素含量
體重指數
糖尿病家族史
年齡

類變數 (0 或 1）
7. For Each Attribute: (all numeric-valued)
   1. Number of times pregnant
   2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
   3. Diastolic blood pressure (mm Hg)
   4. Triceps skin fold thickness (mm)
   5. 2-Hour serum insulin (mu U/ml)
   6. Body mass index (weight in kg/(height in m)^2)
   7. Diabetes pedigree function
   8. Age (years)
   9. Class variable (0 or 1)
```
```
測試資料集
!wget https://raw.githubusercontent.com/MyDearGreatTeacher/AI201909/master/data/pima_data.csv

```
# LogisticRegression()
```
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# 導入數據
filename = 'pima_data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

data = read_csv(filename, names=names)

# 將資料分為輸入資料和輸出結果
array = data.values
X = array[:, 0:8]
Y = array[:, 8]


num_folds = 10
seed = 7

kfold = KFold(n_splits=num_folds, random_state=seed)

model = LogisticRegression()

result = cross_val_score(model, X, Y, cv=kfold)

print(result.mean())
```

# KNN
```
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


# 導入數據
filename = 'pima_data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

data = read_csv(filename, names=names)


# 將資料分為輸入資料和輸出結果
array = data.values

X = array[:, 0:8]
Y = array[:, 8]


num_folds = 10
seed = 7

kfold = KFold(n_splits=num_folds, random_state=seed)

model = KNeighborsClassifier()

result = cross_val_score(model, X, Y, cv=kfold)

print(result.mean())
```
# SVC
```
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

# 導入數據
filename = 'pima_data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

data = read_csv(filename, names=names)

# 將資料分為輸入資料和輸出結果
array = data.values

X = array[:, 0:8]
Y = array[:, 8]


num_folds = 10
seed = 7

kfold = KFold(n_splits=num_folds, random_state=seed)


model = SVC()

result = cross_val_score(model, X, Y, cv=kfold)

print(result.mean())
```
# naive_bayes GaussianNB
```
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

# 導入數據
filename = 'pima_data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

data = read_csv(filename, names=names)

# 將資料分為輸入資料和輸出結果
array = data.values

X = array[:, 0:8]
Y = array[:, 8]


num_folds = 10
seed = 7

kfold = KFold(n_splits=num_folds, random_state=seed)


model = GaussianNB()

result = cross_val_score(model, X, Y, cv=kfold)

print(result.mean())
```
# LDA==LinearDiscriminantAnalysis()
```
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# 導入數據
filename = 'pima_data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

data = read_csv(filename, names=names)


# 將資料分為輸入資料和輸出結果
array = data.values

X = array[:, 0:8]
Y = array[:, 8]


num_folds = 10
seed = 7

kfold = KFold(n_splits=num_folds, random_state=seed)

model = LinearDiscriminantAnalysis()

result = cross_val_score(model, X, Y, cv=kfold)

print(result.mean())
```
# DecisionTreeClassifier
```
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

# 導入數據
filename = 'pima_data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

data = read_csv(filename, names=names)

# 將資料分為輸入資料和輸出結果
array = data.values

X = array[:, 0:8]
Y = array[:, 8]


num_folds = 10
seed = 7
kfold = KFold(n_splits=num_folds, random_state=seed)

model = DecisionTreeClassifier()

result = cross_val_score(model, X, Y, cv=kfold)

print(result.mean())
```
