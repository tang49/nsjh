#
```
[1]Pandas兩大資料型態: Series與DataFrame
[2]Pandas檔案讀寫
[3]Pandas 的運算
```
### 教科書
```
Python資料分析 第二版
Python for Data Analysis, 2nd Edition
作者： Wes McKinney  
譯者： 張靜雯
歐萊禮出版社   出版日期：2018/10/03

https://github.com/wesm/pydata-book

第一章 寫在前面
第二章 Python基礎、IPython 和Jupyter notebook
第三章 內建資料結構、函式和檔案
第四章 NumPy基礎：陣列和向量化計算

第五章 使用pandas
第六章 資料載入、儲存和檔案格式
第七章 資料整理和前處理
第八章 資料處理：連接、合併和重塑
第九章 繪圖與視覺化
第十章 資料聚合和分組
第十一章 時間序列
第十二章 pandas進階
第十三章 Python中的建模函式庫
第十四章 資料分析範例
附錄A 深入NumPy
附錄B 關於IPython系統
```
# [1]Pandas兩大資料型態: Series與DataFrame
```
教科書  第五章 使用pandas
```
```
import pandas as pd

from pandas import Series, DataFrame

import numpy as np
np.random.seed(12345)
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(10, 6))
PREVIOUS_MAX_ROWS = pd.options.display.max_rows
pd.options.display.max_rows = 20
np.set_printoptions(precision=4, suppress=True)
```
## Series資料型態
```
obj = pd.Series([4, 7, -5, 3])
obj

obj.values
obj.index  # like range(4)

obj2 = pd.Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
obj2
obj2.index
```
## DataFrame資料型態
```
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002, 2003],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
frame = pd.DataFrame(data)

frame

frame.head()

pd.DataFrame(data, columns=['year', 'state', 'pop'])

frame2 = pd.DataFrame(data, columns=['year', 'state', 'pop', 'debt'],
                      index=['one', 'two', 'three', 'four',
                             'five', 'six'])
frame2
frame2.columns

frame2['state']
frame2.year

frame2.loc['three']

frame2['debt'] = 16.5
frame2
frame2['debt'] = np.arange(6.)
frame2
```

### DataFrame屬性與方法
```
pandas 有一些好用的屬性與方法可以快速暸解一個 DataFrame 的外觀與內容：

df.shape：這個 DataFrame 有幾列有幾欄
df.columns：這個 DataFrame 的變數資訊
df.index：這個 DataFrame 的列索引資訊
df.info()：關於 DataFrame 的詳細資訊
df.describe()：關於 DataFrame 各數值變數的描述統計
```
# [2]Pandas檔案讀寫的函數

## 抓取遠方的CSV檔案到Google Colab
```
!curl -O https://raw.githubusercontent.com/jakevdp/data-CDCbirths/master/births.csv

!ls

births = pd.read_csv('births.csv')

births.head()

```
## 讀入 csv 文字檔
```
import pandas as pd

# 讀入 csv 文字檔
csv_file = "https://storage.googleapis.com/learn_pd_like_tidyverse/gapminder.csv"

gapminder = pd.read_csv(csv_file)
print(type(gapminder))
gapminder.head()
```

## 讀入 excel 試算表
```
xlsx_file = "https://storage.googleapis.com/learn_pd_like_tidyverse/gapminder.xlsx"
gapminder = pd.read_excel(xlsx_file)
print(type(gapminder))

gapminder.head()
gapminder.tail(5)

gapminder.info()
gapminder.describe()
```
## 上傳檔案到Google Colab再執行
```
from google.colab import files
uploaded = files.upload()
```
## 用 Python 抓取 Ubike 開放資料
```
https://jerrynest.io/python-get-ubike-opendata/

https://colab.research.google.com/drive/1jQprW8RIsA_SEFpBm6POF8DgZ1XZ9YH6#scrollTo=A8qmjnFmGyNv
```
```
Ubike 資料說明

每筆資料有以下 14 個欄位，其中有部分是不會變動的資料，如 sno、sna、sarea 等等，待會放資料庫時會獨立出來放

sno：站點代號
 sna：場站名稱(中文)
 tot：場站總停車格
 sbi：場站目前車輛數量
 sarea：場站區域(中文)
 mday：資料更新時間
 lat：緯度
 lng：經度
 ar：地(中文)
 sareaen：場站區域(英文)
 snaen：場站名稱(英文)
 aren：地址(英文)
 bemp：空位數量
 act：全站禁用狀態
```
```
import requests
import json

url = "http://data.taipei/youbike"
data = requests.get(url).json()

print(data)

for key, value in data["retVal"].items():
  sno = value["sno"]
  sna = value["sna"]
  print("NO.", sno, sna)
```
# ＤataFrame的運算
```
第八章 資料處理：連接、合併和重塑
第十章 資料聚合和分組
```

### pandas裡有幾種方法可以合併資料:以merge()為例
```
Database-Style DataFrame Joins（資料庫風格的DataFrame Joins）
Merge或join操作，能通過一個或多個key，把不同的資料集的行連接在一起。
這種操作主要集中於關聯式資料庫。
pandas中的merge函數是這種操作的主要切入點：
```
```
import pandas as pd
import numpy as np

df1 = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
                    'data1': range(7)})

df1

df2 = pd.DataFrame({'key': ['a', 'b', 'd'],
                    'data2': range(3)})
df2

pd.merge(df1, df2)
```
### GroupBy Mechanics（分組機制）
```
把一個pandas對象（series或DataFrame）按key分解為多個
計算組的匯總統計值（group summary statistics），比如計數，平均值，標準差，或使用者自己定義的函數
```

```
df = pd.DataFrame({'key1' : ['a', 'a', 'b', 'b', 'a'],
                   'key2' : ['one', 'two', 'one', 'two', 'one'],
                   'data1' : np.random.randn(5),
                   'data2' : np.random.randn(5)})
df

grouped = df['data1'].groupby(df['key1'])
grouped
grouped.mean()
```
