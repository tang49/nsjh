

# NUMPY課程大綱
```
[1]numpy
[2]Numpy ndarray資料結構及其屬性
[3]單一Numpy ndarray的各種運算
   建立ndarray的各種方法
   ndarray的各種shape運算:reshape|ravel()|T|newaxis
   ndarray的索引(index)與切片運算(slice)
   ndarray的Reduction 運算:prod()|sum()|mean()|median()
[4]超重要的Universal Functions:Fast Element-Wise Array Functions

[5]多Numpy ndarray的各種運算
[6]Numpy的廣播機制(Broadcast)

[6]應用範例:類神經網路與邏輯閘實作[請參閱另外簡報]

[7]自主學習主題
```
# [1]numpy
```
NumPy是Python語言的一個擴充程式庫。
支援高階大量的維度陣列與矩陣運算
針對陣列運算提供大量的數學函式函式庫。

NumPy的前身Numeric最早是由Jim Hugunin與其它協作者共同開發
2005年，Travis Oliphant在Numeric中結合了另一個同性質的程式庫Numarray的特色，
並加入了其它擴充功能而開發了NumPy。
NumPy為開放原始碼並且由許多協作者共同維護開發。
```
# [2]Numpy ndarray資料結構及其屬性
### NUMPY ndarray(N-Dimensional Arrays)重要屬性:shape dimension
```
軸 (axis) 與維度 (dimension)
shape
dtype 屬性
```
```
import numpy as np
ar2=np.array([[0,3,5],[2,8,7]])
# ar2.shape
ar2.ndim
```
### ndarray資料型態(dtype)與型態轉換(astype)
```
ar=np.array([2,4,6,8]); 
ar.dtype
```
### 下列程式執行後的結果為何?
```
f_ar = np.array([13,-3,8.88])
f_ar

intf_ar=f_ar.astype(int)
intf_ar
```
# [3]單一Numpy ndarray的各種運算
### 建立array(陣列)的方式與實作(有許多方式)
```
1.使用Python內建的array()建立陣列
2.使用numpy提供的創建函數建立陣列
3.直接生成==>使用genfromtxt()方法建立陣列[進階,初學可略過]
```
## 1.使用Python內建的array()建立陣列
```
import numpy as np
x = np.array([[1,2.0],[0,0],(1+1j,3.)])
```
## 2.使用numpy提供的創建函數建立陣列
```
eye
zeros
ones
linspace
indices
diag
tile
```
```
import numpy as np
ar9 = np.eye(3);
ar9
```

### 使用np.zeros建立陣列
```
import numpy as np
np.zeros((2, 3))
```
### 使用np.ones建立陣列
```
import numpy as np
np.ones((4, 7))
```
### 使用np.arange建立陣列
```
import numpy as np
np.arange(2, 3, 0.1) # start, end, step
```
### 使用np.linspace建立陣列
```
import numpy as np
np.linspace(1., 4., 6) # start, end, num
```
### 使用np.linspace建立陣列
```
import numpy as np
np.linspace(2.0, 3.0, num=5)
```
### 使用np.linspace建立陣列
```
import numpy as np
np.linspace(2.0, 3.0, num=5, endpoint=False)
```
### 使用np.linspace建立陣列
```
import numpy as np
np.linspace(2.0, 3.0, num=5, retstep=True)
```
### 使用np.indices建立陣列
```
import numpy as np
np.indices((3, 3))
```
### 使用np.diag建立陣列
```
import numpy as np
ar10=np.diag((2,1,4,6));
ar10
```
```
import numpy as np
a = np.array([0, 1, 2])
np.tile(a, 2)
```
```
import numpy as np
a = np.array([0, 1, 2])
np.tile(a, (2, 2))
```
```
import numpy as np
a = np.array([0, 1, 2])
np.tile(a, (2, 1, 2))
```

## 3.直接生成使用genfromtxt()方法建立陣列[進階,初學可略過]
```
import csv
import numpy as np

x = '''1,3,2,3,1,2,3,4
2,4,5,0.6,5,6,7,8
3,7,8,9,9,10,11,12
4,1,1.1,1.2,13,14,15,16
'''
with open("abc.txt",mode="w",encoding="utf-8") as file:
  file.write(x)
file.close()

np.genfromtxt('abc.txt', delimiter=',', invalid_raise = False)
```

### 下列程式執行後的結果為何?
```
import numpy as np
np.tile(np.array([[1,2],[6,7]]),3)
```
```
import numpy as np
np.tile(np.array([[1,2],[6,7]]),(2,2))
```
### 下列程式執行後的結果為何?
```
import numpy as np
np.array([range(i, i + 3) for i in [2, 4, 6]])
```
```
import numpy as np
np.tile(np.array([[1,2],[6,7]]),3)
```
```
import numpy as np
np.tile(np.array([[1,2],[6,7]]),(2,2))
```
# NUMPY ndarray 運算(Array shape manipulation)
```
reshape
ravel()
T
newaxis
```
### reshape
```
import numpy as np
x = np.arange(2,10).reshape(2,4)
x
```
```
import numpy as np
y = np.arange(2,10).reshape(4,2)
y
```

### ravel
```
import numpy as np
ar=np.array([np.arange(1,6),np.arange(10,15)]); 
ar
```
```
import numpy as np
ar=np.array([np.arange(1,6),np.arange(10,15)])
ar.ravel()
```
### T
```
import numpy as np
ar=np.array([np.arange(1,6),np.arange(10,15)])
ar.T
```
### newaxis
```
import numpy as np
ar=ar[:, np.newaxis]; ar.shape
ar
```
### 下列程式執行後的結果為何?
```
import numpy as np
ar=np.array([np.arange(1,6),np.arange(10,15)])
ar.T.ravel()
```
```
import numpy as np
ar=np.array([14,15,16])
ar=ar[:, np.newaxis]
ar.shape
```
## ndarray的索引(index)與切片運算(slice)
### 1.索引(index)
```
import numpy as np
x = np.arange(2,10)
# x
print(x[0])
print(x[-2])
print(x[-1])
```
```
import numpy as np
ar = np.array([[2,3,4],[9,8,7],[11,12,13]])
ar[1,2]
```
### 2.切片運算(slice)
```
import numpy as np
ar=2*np.arange(6)
# ar
ar[1:5:2]
```
```
import numpy as np
ar=2*np.arange(6)
# ar
ar[1:6:2]
```
```
import numpy as np
ar=2*np.arange(6)
# ar
ar[:4]
```
### [小測驗]下列程式執行後的結果為何?
```
import numpy as np
ar = np.array([[2,3,4],[9,8,7],[11,12,13]])
# ar
ar[2,:] 
```
```
import numpy as np
ar = np.array([[2,3,4],[9,8,7],[11,12,13]])
# ar
ar[:,1]  
```
```
import numpy as np
ar = np.array([[2,3,4],[9,8,7],[11,12,13]])
# ar
ar[2,-1]
```
```
import numpy as np
ar=2*np.arange(6)
# ar
ar[::3]
```
```
import numpy as np
ar=2*np.arange(6)
# ar
ar[:3]=1
ar
```
```
import numpy as np
ar=2*np.arange(6)
# ar
ar[2:]=np.ones(4)
ar
```
## ndarray的Reduction 運算:prod()|sum()|mean()|median()
```
prod()
sum()
mean()
median(ar)
```
###
```
import numpy as np
ar=np.arange(1,5)
ar.prod()
```
```
import numpy as np
ar=np.array([[2,3,4],[5,6,7],[8,9,10]])
ar.sum()
```
```
import numpy as np
ar=np.array([[2,3,4],[5,6,7],[8,9,10]])
ar.mean()
```
```
import numpy as np
ar=np.array([[2,3,4],[5,6,7],[8,9,10]])
np.median(ar)
```
### [小測驗]下列程式執行後的結果為何?
```
ar=np.array([np.arange(1,6),np.arange(1,6)])
# ar
np.prod(ar,axis=0)
```
```
ar=np.array([np.arange(1,6),np.arange(1,6)])
# ar
np.prod(ar,axis=1)
```
### 延伸閱讀
```
更多運算請參閱
NumPy 高速運算徹底解說 - 
六行寫一隻程式？你真懂深度學習？手工算給你看！ 現場で使える! NumPyデータ処理入門
吉田 拓真、尾原 颯 著 吳嘉芳、蒲宗賢 譯
旗標科技2020-01-17
Ch02 NumPy基本運算函式

Ch03 NumPy 的數學函式
```

# [5]超重要的ufunc(Universal Functions):Fast Element-Wise Array Functions
```
ufunc是universal function的縮寫
這些函數能夠作用於narray物件的每一個元素上，而不是針對narray物件操作
numpy提供了大量的ufunc的函數。這些函數在對narray進行運算的速度比使用迴圈或者清單推導式要快很多

sqrt
exp

```
### sqrt
```
import numpy as np
arr = np.arange(10)
np.sqrt(arr)
```
### exp
```
import numpy as np
arr = np.arange(10)
np.exp(arr)
```


## [5]多Numpy ndarray的各種運算(A矩陣與B矩陣間的運算)
### 加法運算
```
import numpy as np
arr = np.array([[1., 2., 3.], [4., 5., 6.]])
# arr
arr+arr
```
### 乘法運算
```
import numpy as np
arr = np.array([[1., 2., 3.], [4., 5., 6.]])
# arr
arr*arr
```
```
import numpy as np
ar=np.array([[1,1],[1,1]])
ar2=np.array([[2,2],[2,2]])
ar*ar2
```
### dot運算
```
import numpy as np
ar=np.array([[1,1],[1,1]])
ar2=np.array([[2,2],[2,2]])
ar.dot(ar2)
```
### [小測驗]下列程式執行後的結果為何?
```
import numpy as np
arr = np.array([[1., 2., 3.], [4., 5., 6.]])
# arr
arr-arr
```
```
import numpy as np
arr = np.array([[1., 2., 3.], [4., 5., 6.]])
# arr
1/arr
```
```
import numpy as np
arr = np.array([[1., 2., 3.], [4., 5., 6.]])
# arr
arr ** 0.5
```
```
import numpy as np
xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])
result = [(x if c else y)
          for x, y, c in zip(xarr, yarr, cond)]
result
```
```
import numpy as np
xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])
result = np.where(cond, xarr, yarr)
result
```
## [6]ndarray 運算的Broadcasting(廣播機制)
```
參考資料:
[1]numpy 中的 broadcasting（廣播）機制
   https://codertw.com/程式語言/462108/
[2]NumPy 廣播(Broadcast)
   https://www.runoob.com/numpy/numpy-broadcast.html[2]
```
```
import numpy as np
x1 = np.arange(9.0).reshape((3, 3))
x2 = np.arange(3.0)
np.multiply(x1, x2)
```
```
import numpy as np
np.arange(3) + 5
```
```
import numpy as np
np.ones((3, 3)) + np.arange(3)
```
### [小測驗]下列程式執行後的結果為何?
```
import numpy as np
np.arange(3).reshape((3, 1)) + np.arange(3)
```
# 
# [7]自主學習主題
## 使用<Numpy random模組>產生隨機資料
```
numpy.random.randint(low, high=None, size=None, dtype='l')

seed
random.randint

```
```
import numpy as np
np.random.randint(1,5)
```
```
import numpy as np
np.random.randint(-5,5,size=(2,2))
```
```
import numpy as np
np.random.seed(0)
x1 = np.random.randint(10, size=6) 
x1
```
```
import numpy as np
np.random.seed(0)
x2 = np.random.randint(10, size=(3, 4))
x2
```
```
import numpy as np
np.random.seed(0)
x3 = np.random.randint(10, size=(3, 4, 5))
x3
```
### [小測驗]下列程式執行後的結果為何?
```
import numpy as np
np.random.randint(1,size=5)
```
## NUMPY ndarray 運算(A矩陣與B矩陣間的convolute運算)[進階,初學可略過]
```
numpy.convolve(a, v, mode=‘full’ )

convolve
```
```
import numpy as np
np.convolve([1, 2, 3], [0, 1, 0.5])
```
```
import numpy as np
np.convolve([1,2,3],[0,1,0.5], 'same')
```
```
import numpy as np
np.convolve([1,2,3],[0,1,0.5], 'valid')
```
### [小測驗]下列程式執行後的結果為何?
```
import numpy as np
np.convolve([1, 2, 3], [0, 1, 0.5])
```
## NUMPY ndarray(N-Dimensional Arrays)檔案輸入與輸出
```
https://ithelp.ithome.com.tw/articles/10196167
save()
load()
```
## NUMPY ndarray 運算 - 排序sort[進階,初學可略過]
```
https://github.com/femibyte/mastering_pandas/blob/master/MasteringPandas-chap3_DataStructures.ipynb

學生報告:舉例說明numpy陣列的各項排序運算

sort
```
```
import numpy as np
ar=np.array([[3,2],[10,-1]])
# ar
ar.sort(axis=1)
ar
```
### [小測驗]下列程式執行後的結果為何?
```
import numpy as np
ar=np.array([[3,2],[10,-1]])
# ar
ar.sort(axis=0)
ar
```
```
import numpy as np
ar=np.random.randint(10,size=5)
ar.sort()
ar[::-1]
```

# 龍大大 我的億 大樂透 開獎中心
```
說明底下程式的處理邏輯
```
```
import random as rand

list1 = rand.sample(range(1,50), 7)

special = list1.pop()

list1.sort()

print("龍大大 我的億 大樂透 開獎中心")
print("本期大樂透中獎號碼為：", end="")
for i in range(0,6):
    if i == 5:    print(str(list1[i]))
    else:    print(str(list1[i]), end=", ")
print("本期大樂透特別號為：" + str(special))
```
```
說明底下程式的處理邏輯

[1]產生從1到49的7個不重複的整數

[2]設定中獎號法
挑出一個當作 特別號
剩下六個當作 中獎號碼

[3]印出 特殊格式輸出 的 中獎單
```
