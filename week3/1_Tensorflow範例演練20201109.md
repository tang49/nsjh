# 推薦參考書
```
Towards Tensorflow 2.0：無痛打造AI模型（iT邦幫忙鐵人賽系列書）
陳峻廷  博碩文化 2020-07-08
```

###
```
# -*- coding: utf-8 -*-


import tensorflow as tf
tf.__version__
```
## Tensor ndarray  matrix
### 使用tf.constant建立各種常數Tensor
```
tf.constant(123)

tf.constant(3.3)

tf.constant([True,False])

tf.constant('I love tf2!!!')
```
```
import numpy as np

# 建立 tf.constant
a = tf.constant([11,121,13])

#Convert np arry to tf
a = np.array([11,121,13])
b = tf.convert_to_tensor(a)

b
```
### 建立變數 tf.Variable
```
a = tf.ones(6)

b = tf.Variable(a,name='Data_point')
b
```
### 建立 matrix==>Tensor
```
a = tf.zeros([6,6])
a
```
```
# 複製
b = tf.zeros_like(a)
b
```
```
# 建立不同數值 or fill -> shape and fill nums
c = tf.fill([6,6],1)

c
```
### 根據機率分布建立tensor
```
# 正則分布
a = tf.random.normal([6,6],mean=0,stddev=1)
a
```
```
#random by truncated(截斷型)normal distribution
b = tf.random.truncated_normal([6,6],mean=0,stddev=1)
a
```
```
# 均勻分布
c = tf.random.uniform([6,6],minval=0,maxval=1)
c
```

## Tensor的簡單運算
```
a = tf.ones([1,5,5,3])
a
a[0][0]
a[0][0].shape
```
```
a[...,2].shape
#Output shape:TensorShape([1, 5, 5])
```

### Tensor的排序運算Sorting
```
data = tf.random.normal([10],mean=0,stddev=1)
data
tf.sort(data,direction='DESCENDING')
```
```
#
top_data = tf.math.top_k(data,k=5)

#
top_data.indices
#or
top_data.values
```

### tensor運算pad,clip
```
data = tf.random.normal([3,3],mean=0,stddev=1)
data
tf.pad(data,[[1,1],[1,1]])

#Example clipping
tf.clip_by_value(data,0,1)

#range if number less or above fix at 0 or 1

tf.clip_by_value(data,0,1)
```

## tensor與tensor運算
```
# element-wise
a = tf.fill([2,2],5)
b = tf.fill([2,2],6)
a+b,a*b
```
```
# matrix-wise
a = tf.fill([2,32,4],5)
b = tf.fill([2,4,6],6)
tf.matmul(a,b)
#Output:TensorShape([2, 32, 6])
```
# 線性回歸
```
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

# Parameters.
learning_rate = 0.1  #學習率
training_steps = 1000 #訓練回數
display_step = 100 #顯示回數
n_samples = 50 #樣本數

# 產生模擬數字
X = np.random.rand(n_samples).astype(np.float32)
Y = X * 10 + 5

W = tf.Variable(tf.random.normal([1]))
b = tf.Variable(tf.zeros([1]))
```

```
#  定義最終的線性回歸方程式(Wx + b).
def linear_regression(x):
    return W * x + b

# 定義誤差Mean square error.
def mean_square(y_pred, y_true):
    return tf.reduce_sum(tf.pow(y_pred-y_true, 2)) / ( n_samples)

# 使用Stochastic Gradient Descent Optimizer.
optimizer = tf.optimizers.SGD(learning_rate)

# 定義執行流程Optimization process. 
def run_optimization():
    # Wrap computation inside a GradientTape for automatic differentiation.
    
    with tf.GradientTape() as g:
        pred = linear_regression(X)
        loss = mean_square(pred, Y)

    # Compute gradients.
    gradients = g.gradient(loss, [W, b])
    
    # Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, [W, b]))
```
```
# 執行
for step in range(1, training_steps + 1):
    # Run the optimization to update W and b values.
    run_optimization()
    
    if step % display_step == 0:
        pred = linear_regression(X)
        loss = mean_square(pred, Y)
        print("step: %i, loss: %f, W: %f, b: %f" % (step, loss, W.numpy(), b.numpy()))
```
## 畫圖
```
import matplotlib.pyplot as plt

# Graphic display
plt.plot(X, Y, 'ro', label='Original data')
plt.plot(X, np.array(W * X + b), label='Fitted line')
plt.legend()
plt.show()
```

