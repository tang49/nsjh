# 教學大綱
```
[1].Data Visualization資料視覺化
[2].資料視覺化の套件
[3].Google Colab上的範利

[4].MATPLOTLIB
[5].MATPLOTLIB範例學習[1]單一圖形
    matplotlib.pyplot的許多範例
     plot():折線圖(Line chart):matplotlib.pyplot.plot
     bar():長條圖|柱狀圖(Bar Chart):matplotlib.pyplot.bar
     hist():直方圖(histogram):matplotlib.pyplot.hist
     boxplot():箱形圖 (Box plot):matplotlib.pyplot.boxplot
     scatter():散佈圖 (Scatter plot): matplotlib.pyplot.scatter
     圓餅圖
[6].MATPLOTLIB範例學習[2]多圖形並陳
```
# [1].Data Visualization資料視覺化
```
藉助於圖形化手段，
清晰有效地傳達與溝通訊息

https://zh.wikipedia.org/wiki/資料視覺化
```
# [2].資料視覺化の套件
```
Matp2lotlib(本課程使用)
Seaborn
Ggplot
Bokeh
Pyga
Plotly
```
# [3].Google Colab上的範利
```
Charting in Colaboratory
https://colab.research.google.com/notebooks/charts.ipynb
```
### Line Plots折線圖:基本統計圖形
```
import matplotlib.pyplot as plt
 
x  = [1, 2, 3, 4, 5, 6, 7, 8, 9]
y1 = [1, 3, 5, 3, 1, 3, 5, 3, 1]
y2 = [2, 4, 6, 4, 2, 4, 6, 4, 2]
plt.plot(x, y1, label="line L")
plt.plot(x, y2, label="line H")
plt.plot()

plt.xlabel("x axis")
plt.ylabel("y axis")
plt.title("Line Graph Example")
plt.legend()
plt.show()
```

### [隨堂小測驗]Line Plots折線圖:基本統計圖形
```
顏色改成紅色
線條改成虛線
```
```
plt.plot(x, y1,'r--',label="line L")
```
```
https://matplotlib.org/3.3.1/api/_as_gen/matplotlib.pyplot.plot.html
```
# [4].MATPLOTLIB
```
官方網址 https://matplotlib.org/

使用指南  https://matplotlib.org/users/index.html

學習指南(Tutorials) https://matplotlib.org/tutorials/index.html
```
```
https://zh.wikipedia.org/wiki/Matplotlib
https://blog.techbridge.cc/2018/05/11/python-data-science-and-machine-learning-matplotlib-tutorial/
https://www.runoob.com/numpy/numpy-matplotlib.html
```
# [5]MATPLOTLIB範例學習[1]單一圖形
## matplotlib.pyplot
```
https://matplotlib.org/api/pyplot_summary.html
```
```
matplotlib.pyplot模組有許多基本統計圖形的函數
plot():折線圖:matplotlib.pyplot.plot
pie():
bar():長條圖|柱狀圖(Bar Chart):matplotlib.pyplot.bar
hist():直方圖(histogram):matplotlib.pyplot.hist
boxplot():箱形圖 (Box plot):matplotlib.pyplot.boxplot
scatter():散佈圖 (Scatter plot): matplotlib.pyplot.scatter


參考: https://colab.research.google.com/notebooks/charts.ipynb
```

### plot():折線圖範例1
```
import numpy as np
import pylab as pl


# 產生資料
x = np.arange(0.0, 2.0*np.pi, 0.01)	
y = np.sin(x)			

#畫圖

pl.plot(x,y)		
pl.xlabel('x')			
pl.ylabel('y')
pl.title('sin')		
pl.show()
```
```
步驟一:先產生x軸的資料===使用陣列:0到2π之間，以0.01為step
x = np.arange(0.0, 2.0*np.pi, 0.01)    
 
步驟二:針對每一個x產生 y (y = sin(x))==== y 也是一個陣列
 
y = np.sin(x)

步驟三:畫圖==>設定圖形的呈現參數
pl.plot(x,y)	

步驟四:顯示圖形
pl.show()
```

## plot()函數 matplotlib.pyplot.plot
```
https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot

使用plot()函數畫圖
pl.plot(t,s)            #畫圖，以t為橫坐標，s為縱坐標
pl.xlabel('x')            #設定坐標軸標籤
pl.ylabel('y')
pl.title('sin')        #設定圖形標題
pl.show()                #顯示圖形
```
### plot():折線圖範例2:看看底下產生的數學公式
```
https://matplotlib.org/gallery/pyplots/pyplot_mathtext.html#sphx-glr-gallery-pyplots-pyplot-mathtext-py
```
```
import numpy as np
import matplotlib.pyplot as plt
t = np.arange(0.0, 2.0, 0.01)
s = np.sin(2*np.pi*t)

plt.plot(t, s)
plt.title(r'$\alpha_i > \beta_i$', fontsize=20)

plt.text(1, -0.6, r'$\sum_{i=0}^\infty x_i$', fontsize=20)
plt.text(0.6, 0.6, r'$\mathcal{A}\mathrm{sin}(2 \omega t)$',
         fontsize=20)
         
plt.xlabel('time (s)')
plt.ylabel('volts (mV)')
plt.show()
```
## 圓餅圖(Pie Chart)
```
https://zh.wikipedia.org/wiki/%E9%A5%BC%E5%9B%BE
圓餅圖，或稱餅狀圖，是一個劃分為幾個扇形的圓形統計圖表，
用於描述量、頻率或百分比之間的相對關係。

在圓餅圖中，每個扇區的弧長（以及圓心角和面積）大小為其所表示的數量的比例。
這些扇區合在一起剛好是一個完全的圓形。
顧名思義，這些扇區拼成了一個切開的餅形圖案。
```
### 圓餅圖(Pie Chart)範例學習
```
#設定label：
labels = 'A','B','C','D','E','F'

#設定每個區塊的大小：
size = [33,52,12,17,62,48]

# 使用plt.pie()畫圓餅圖

plt.pie(size , labels = labels,autopct='%1.1f%%')

# autopct='%1.1f%%'是用來顯示百分比。

#為了要讓圓餅圖比例相等加上：
plt.axis('equal')

# 最後的顯示
plt.show()
```
## 直方圖
```
https://zh.wikipedia.org/wiki/直方圖

直方圖基本上是一種次數分配表，
沿著橫軸以各組組界為分界，組距為底邊，以各組的次數為高度，
依序在固定的間距上畫出矩形高度所繪製而成之圖形。
```
```
import matplotlib.pyplot as plt
from numpy.random import normal,rand

x = normal(size=200)

plt.hist(x,bins=30)
plt.show()
```
## 長條圖|柱狀圖(Bar Chart)
```
長條圖（英語：bar chart），亦稱條圖（英語：bar graph）、條狀圖、棒形圖、柱狀圖、條形圖表、條形圖，
是一種以長方形的長度為變量的統計圖表。
長條圖用來比較兩個或以上的價值（不同時間或者不同條件），只有一個變量，
通常利用於較小的數據集分析。
長條圖亦可橫向排列，或用多維方式表達。

繪製長條圖時，長條柱或柱組中線須對齊項目刻度。
相較之下，折線圖則是將數據代表之點對齊項目刻度。
在數字大且接近時，兩者皆可使用波浪形省略符號，以擴大表現數據間的差距，增強理解和清晰度。

類似的圖形表達為直方圖，不過後者較長條圖而言更複雜（直方圖可以表達兩個不同的變量）。
```
```
matplotlib.pyplot.bar(left, height, alpha=1, width=0.8, color=, edgecolor=, label=, lw=3)

參數：
1. left：x軸的位置序列，一般採用arange函數產生一個序列；
2. height：y軸的數值序列，也就是直條圖的高度，一般就是我們需要展示的資料；
3. alpha：透明度
4. width：為直條圖的寬度，一般這是為0.8即可；
5. color或facecolor：直條圖填充的顏色；
6. edgecolor：圖形邊緣顏色
7. label：解釋每個圖像代表的含義
8. linewidth or linewidths or lw：邊緣or線的寬度
```
### 長條圖範例一
```
from matplotlib import pyplot as plt 

x =  [5,8,10] 
y =  [12,16,6] 
x2 =  [6,9,11] 
y2 =  [6,15,7] 

plt.bar(x, y, align =  'center') 
plt.bar(x2, y2, color =  'g', align =  'center') 

plt.title('Bar graph') 
plt.ylabel('Y axis') 
plt.xlabel('X axis') 
plt.show()
```
### 長條圖範例二
```
%matplotlib inline
import numpy as np
from matplotlib import pyplot as plt

plt.figure(figsize=(9,6))

n = 8
X = np.arange(n)+1 #X是1,2,3,4,5,6,7,8,柱的個數
#uniform均勻分佈的亂數，normal是正態分佈的亂數，0.5-1均勻分佈的數，一共有n個
Y1 = np.random.uniform(0.5,1.0,n)
Y2 = np.random.uniform(0.5,1.0,n)

plt.bar(X, Y1, alpha=0.9, width = 0.35, facecolor = 'lightskyblue', edgecolor = 'white', label='one', lw=1)
plt.bar(X+0.35, Y2, alpha=0.9, width = 0.35, facecolor = 'yellowgreen', edgecolor = 'white', label='second', lw=1)
plt.legend(loc="upper left") # label的位置在左上，沒有這句會找不到label去哪了
```
## 散佈圖 (Scatter plot)  
```
用途:看看資料有何關係??
```
```
https://en.wikipedia.org/wiki/Scatter_plot

散佈圖是一種使用笛卡兒坐標來顯示一組數據的通常兩個變量的值的圖或數學圖。
如果對點進行了編碼，則可以顯示一個附加變量。
數據顯示為點的集合，每個點具有確定水平軸上位置的一個變量的值和確定垂直軸上位置的另一個變量的值。
```

### 散佈圖 (Scatter plot)範例一
```
import numpy as np
import pylab as pl

# 產生資料
x = np.arange(0, 2.0*np.pi, 0.2)
y = np.cos(x)

#畫圖
pl.scatter(x,y)			
pl.show()
```
```
改成底下公式:
y = np.exp(x)*np.cos(x)
```
### 散佈圖 (Scatter plot)範例二
```
import matplotlib.pylab as pl
import numpy as np

# 產生資料
x = np.random.random(100)
y = np.random.random(100)

#畫圖
#pl.scatter(x,y,s=x*500,c=u'r',marker=u'*')	
pl.scatter(x,y,s=x*500,c=u'b',marker=u'p')	
pl.show()
```
#### matplotlib使用函數:pl.scatter
```
https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.scatter.html
有許多參數設定:請參看原始網站

s指大小，c指顏色，marker指符號形狀
```
### matplotlib.markers符號形狀
```
https://matplotlib.org/api/markers_api.html?highlight=marker
上網看看如何改變markers
```
## boxplot箱形圖 (Box plot)範例
```
https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.boxplot.html

matplotlib.pyplot.boxplot(x, notch=None, sym=None, vert=None, whis=None, 
positions=None, widths=None, patch_artist=None, bootstrap=None, usermedians=None, 
conf_intervals=None, meanline=None, showmeans=None, showcaps=None, showbox=None, 
showfliers=None, boxprops=None, labels=None, flierprops=None, medianprops=None, 
meanprops=None, capprops=None, whiskerprops=None, manage_ticks=True, autorange=False, zorder=None, *, data=None)
```
```
import numpy as np
import matplotlib.pyplot as plt

# Fixing random state for reproducibility
np.random.seed(19680801)

# fake up some data
spread = np.random.rand(50) * 100
center = np.ones(25) * 50
flier_high = np.random.rand(10) * 100 + 100
flier_low = np.random.rand(10) * -100
data = np.concatenate((spread, center, flier_high, flier_low))

fig1, ax1 = plt.subplots()
ax1.set_title('Basic Plot')
ax1.boxplot(data)
```
### 作業:完成底下的boxplot箱形圖 (Box plot)
```
某高中身高
            178   164  159  162  182  
             179   166  168  173  165
http://estat.ncku.edu.tw/topic/graph_stat/base/BoxPlot.html
```

# 單一圖形顯示多筆資料
```
把很多張圖畫到一個顯示介面
===>面板切分成一個一個子圖
```
```
[Day16]視覺化資料 - Matplotlib - legend、subplot、GridSpec、annotate
https://ithelp.ithome.com.tw/articles/10201670
```
## matplotlib.pyplot.legend的用處
```
import numpy as np
import pylab as pl

# 產生資料
x = np.arange(0.0, 2.0*np.pi, 0.01)
y = np.sin(x)						
z = np.cos(x)						

#畫圖
pl.plot(x, y, label='sin()')
pl.plot(x, z, label='cos()')
pl.xlabel('x')		
pl.ylabel('y')
pl.title('sin-cos') 


pl.legend(loc='center')	
#pl.legend(loc='upper right')	
#pl.legend(loc='upper right')
#pl.legend()				
pl.show()
```
### matplotlib.pyplot.legend
```
語法：legend(*args)
https://ithelp.ithome.com.tw/articles/10201670
https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html#matplotlib.pyplot.legend
```
### 範例練習:帶有數學公式的圖形
```
import numpy as np
import matplotlib.pyplot as plt


# 產生資料
x = np.linspace(0, 2*np.pi, 500)
y = np.sin(x)
#z = np.cos(x*x)
z = np.cos(x*x*x)


#畫圖
plt.figure(figsize=(8,5))
#標籤前後加上$，代表以內嵌的LaTex引擎顯示為公式
plt.plot(x,y,label='$sin(x)$',color='red',linewidth=5)	#紅色，2個像素寬
plt.plot(x,z,'b--',label='$cos(x^2)$')		#b::藍色，--::虛線
plt.xlabel('Time(s)')
plt.ylabel('Volt')
plt.title('Sin and Cos figure using pyplot')
plt.ylim(-1.2,1.2)

plt.legend()							
plt.show()
```

## matplotlib.pyplot.subplot
```
https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.subplot.html
```
```
要先載入套件
import matplotlib.pyplot as plt

方法一：plt.subplot(nrow, ncol, x)===>plt.subplot('行','列','編號')
範例:plt.subplot(2,2,1)或plt.subplot(221)

方法二：
先使用plt.figure()產生圖形

fig = plt.figure()

在一個一個子圖加入
plt.add_subplot(nrow, ncol,x)

相對於上一種方法，此種方法可以引入變數，比如要繪製四個子圖：
for i in range(4):
    fig.add_subplot(4,1,i+1)
```
### subplot範例練習1:多圖合併
```
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0, 100)

plt.subplot(221)
plt.plot(x, x)
#作圖2
plt.subplot(222)
plt.plot(x, -x)


#作圖3
plt.subplot(223)
plt.plot(x, x ** 2)
plt.grid(color='r', linestyle='--', linewidth=1,alpha=0.3)

#作圖4
plt.subplot(224)
plt.plot(x, np.log(x))
plt.show()
````
### subplot範例練習2:多圖合併
```
import numpy as np
import matplotlib.pyplot as plt

# 產生資料
x= np.linspace(0, 2*np.pi, 500)	
y1 = np.sin(x)					
y2 = np.cos(x)
y3 = np.sin(x*x)

#畫圖
plt.figure(1)					#建立圖形
#create three axes
ax1 = plt.subplot(2,2,1)			#第一列第一行圖形
ax2 = plt.subplot(2,2,2)			#第一列第二行圖形
ax3 = plt.subplot(2,1,2)			#第二列

##設定第一張圖
plt.sca(ax1)					#選擇ax1
plt.plot(x,y1,color='red')		#繪製紅色曲線
plt.ylim(-1.2,1.2)				#限制y坐標軸範圍

##設定第二張圖
plt.sca(ax2)					#選擇ax2
plt.plot(x,y2,'b--')			#繪製藍色曲線
plt.ylim(-1.2,1.2)

##設定第三張圖
plt.sca(ax3)					#選擇ax3
plt.plot(x,y3,'g--')
plt.ylim(-1.2,1.2)

##多圖並呈
plt.show()
```

### 圖中圖的技術add_axes()
```
import matplotlib.pyplot as plt

#新建figure
fig = plt.figure()

# 定義資料
x = [1, 2, 3, 4, 5, 6, 7]
y = [1, 3, 4, 2, 5, 8, 6]

#新建區域ax1
#figure的百分比,從figure 10%的位置開始繪製, 寬高是figure的80%
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
# 獲得繪製的控制代碼
ax1 = fig.add_axes([left, bottom, width, height])
ax1.plot(x, y, 'r')
ax1.set_title('area1')

#新增區域ax2,巢狀在ax1內
left, bottom, width, height = 0.2, 0.6, 0.25, 0.25
# 獲得繪製的控制代碼
ax2 = fig.add_axes([left, bottom, width, height])
ax2.plot(x,y, 'b')
ax2.set_title('area2')

plt.show()
```

# 自主學習主題
## 使用GridSpec畫不同比例的多圖形
```
https://www.itread01.com/content/1541685249.html
```
##
```
Matplotlib 畫動態圖 animation模組

https://www.itread01.com/content/1547022071.html
```
### 3D圖形畫製
```
https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html
```
### 3D圖形畫製範例練習:
```
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d


# 產生資料
x,y = np.mgrid[-2:2:20j, -2:2:20j]
z = 50 * np.sin(x+y)		#測試資料

#畫圖
ax = plt.subplot(111, projection='3d')	#三維圖形
ax.plot_surface(x,y,z,rstride=2, cstride=1, cmap=plt.cm.Blues_r)
ax.set_xlabel('X')			#設定坐標軸標籤
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
```
### 延伸閱讀:推薦的教科書plot.ly

```
官方網址https://plot.ly/看看互動式資料視覺化成果
```
```
Python數據分析：基於Plotly的動態可視化繪圖
作者： 孫洋洋, 王碩, 邢夢來, 袁泉, 吳娜
電子工業出版社
https://github.com/sunshe35/PythonPlotlyCodes
```

### 延伸閱讀:書bokeh
```
官方網址  https://bokeh.pydata.org/en/latest/
!pip install bokeh

```
```
import numpy as np
from bokeh.plotting import figure, show
from bokeh.io import output_notebook

N = 4000

x = np.random.random(size=N) * 100
y = np.random.random(size=N) * 100
radii = np.random.random(size=N) * 1.5
colors = ["#%02x%02x%02x" % (r, g, 150) for r, g in zip(np.floor(50+2*x).astype(int), np.floor(30+2*y).astype(int))]

output_notebook()
p = figure()
p.circle(x, y, radius=radii, fill_color=colors, fill_alpha=0.6, line_color=None)
show(p)
```
### lorenz attractor範例
```
https://docs.bokeh.org/en/latest/docs/gallery/lorenz.html
```
```
import numpy as np
from scipy.integrate import odeint

from bokeh.plotting import figure, output_file, show

sigma = 10
rho = 28
beta = 8.0/3
theta = 3 * np.pi / 4

def lorenz(xyz, t):
    x, y, z = xyz
    x_dot = sigma * (y - x)
    y_dot = x * rho - x * z - y
    z_dot = x * y - beta* z
    return [x_dot, y_dot, z_dot]

initial = (-10, -7, 35)
t = np.arange(0, 100, 0.006)

solution = odeint(lorenz, initial, t)

x = solution[:, 0]
y = solution[:, 1]
z = solution[:, 2]
xprime = np.cos(theta) * x - np.sin(theta) * y

colors = ["#C6DBEF", "#9ECAE1", "#6BAED6", "#4292C6", "#2171B5", "#08519C", "#08306B",]

p = figure(title="Lorenz attractor example", background_fill_color="#fafafa")

p.multi_line(np.array_split(xprime, 7), np.array_split(z, 7),
             line_color=colors, line_alpha=0.8, line_width=1.5)

output_file("lorenz.html", title="lorenz.py example")

show(p)
```
#
```
https://matplotlib.org/gallery/lines_bars_and_markers/fill.html#sphx-glr-gallery-lines-bars-and-markers-fill-py
```
```
import numpy as np
import matplotlib.pyplot as plt


def koch_snowflake(order, scale=10):
    """
    Return two lists x, y of point coordinates of the Koch snowflake.

    Arguments
    ---------
    order : int
        The recursion depth.
    scale : float
        The extent of the snowflake (edge length of the base triangle).
    """
    def _koch_snowflake_complex(order):
        if order == 0:
            # initial triangle
            angles = np.array([0, 120, 240]) + 90
            return scale / np.sqrt(3) * np.exp(np.deg2rad(angles) * 1j)
        else:
            ZR = 0.5 - 0.5j * np.sqrt(3) / 3

            p1 = _koch_snowflake_complex(order - 1)  # start points
            p2 = np.roll(p1, shift=-1)  # end points
            dp = p2 - p1  # connection vectors

            new_points = np.empty(len(p1) * 4, dtype=np.complex128)
            new_points[::4] = p1
            new_points[1::4] = p1 + dp / 3
            new_points[2::4] = p1 + dp * ZR
            new_points[3::4] = p1 + dp / 3 * 2
            return new_points

    points = _koch_snowflake_complex(order)
    x, y = points.real, points.imag
    return x, y
```
### 延伸閱讀: seaborn
```
範例學習1:
https://colab.research.google.com/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/04.14-Visualization-With-Seaborn.ipynb
```
```
範例學習2:
https://colab.research.google.com/drive/1o6MijFkNHiTPeS8Y5n59j2cH4-Mf2wX3
```
```
import seaborn as sns
sns.set(style="ticks")

# Load the example dataset for Anscombe's quartet
df = sns.load_dataset("anscombe")

# Show the results of a linear regression within each dataset
sns.lmplot(x="x", y="y", col="dataset", hue="dataset", data=df,
           col_wrap=2, ci=None, palette="muted", height=4,
           scatter_kws={"s": 50, "alpha": 1});
```
```
https://www.data-insights.cn/?p=179
```

### 延伸閱讀:  altair
```
官方網址https://altair-viz.github.io/ 

```
```
import altair as alt
from vega_datasets import data
cars = data.cars()

alt.Chart(cars).mark_point().encode(
    x='Horsepower',
    y='Miles_per_Gallon',
    color='Origin',
).interactive()
```
