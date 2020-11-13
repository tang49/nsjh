# 體驗主題
```
1.圖片分類==>使用MLP
2.圖片分類==>使用CNN
3.使用別人的Model[Google InceptionV3]進行圖片辨識
4.transfer Learning 遷移學習
```
## 1.圖片分類==>使用MLP
```
https://www.tensorflow.org/tutorials/quickstart/beginner?hl=zh-tw
```
```
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
```
### 定義模型
```
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])
```
```
predictions = model(x_train[:1]).numpy()
predictions
```
```
# tf.nn.softmax function converts these logits to "probabilities" for each class:

tf.nn.softmax(predictions).numpy()
```
```
#The losses.SparseCategoricalCrossentropy loss takes a vector of logits and a True index and returns a scalar loss for each example

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```
```
loss_fn(y_train[:1], predictions).numpy()
```
```
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
```

### 使用fit()進行訓練
```
model.fit(x_train, y_train, epochs=5)
```
### 使用evaluate()進行預測
```
model.evaluate(x_test,  y_test, verbose=2)
```

# 2.圖片分類==>使用CNN
```
https://www.tensorflow.org/tutorials/images/cnn?hl=zh-tw
```
```

```
# 3.使用別人的Model[Google InceptionV3]進行圖片辨識
```
!wget https://images.freeimages.com/images/large-previews/0cd/mango-1327290.jpg
```
```
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

## 使用Google InceptionV3模型
model = tf.keras.applications.InceptionV3(include_top=True, weights='imagenet')

## 看看Google InceptionV3模型的結構
model.summary()

from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications.inception_v3 import decode_predictions


# 撰寫讀取圖片的函數
def read_img(img_path, resize=(299,299)):
    img_string = tf.io.read_file(img_path)  # 讀取檔案
    img_decode = tf.image.decode_image(img_string)  # 將檔案以影像格式來解碼
    img_decode = tf.image.resize(img_decode, resize)  # 將影像resize到網路輸入大小
    # 將影像格式增加到4維(batch, height, width, channels)，模型預測要求格式
    img_decode = tf.expand_dims(img_decode, axis=0)
    return img_decode

# 
img_path = 'mango-1327290.jpg' # 要辨識的圖片

img = read_img(img_path) #讀取圖片

img = preprocess_input(img)  # 圖片前處理

preds = model.predict(img)  # 預測圖片

print("Predicted:", decode_predictions(preds, top=3)[0])  # 輸出預測最高的三個類別
```
## 4.transfer Learning 遷移學習
```
https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub?hl=zh-tw
```
```

```
