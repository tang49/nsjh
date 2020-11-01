#
```


```

# 線性迴歸
##
```
<html>
<head>
<title>Ch15_1_2.html</title>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"></script>
</head>
<body>
<script>

// 建立線性迴歸模型
const model = tf.sequential();
model.add(tf.layers.dense({units: 16, inputShape: [1]}));
model.add(tf.layers.dense({units: 1}));
model.compile({loss:"meanSquaredError", optimizer:"adam", metrics:["mse"]});

// 訓練資料
const xs = tf.tensor2d([29,28,34,31,25,29,32,
                        31,24,33,25,31,26,30], [14,1]);
const ys = tf.tensor2d([7.7,6.2,9.3,8.4,5.9,6.4,8.0,
                        7.5,5.8,9.1,5.1,7.3,6.5,8.4], [14,1]);


// 訓練模型
model.fit(xs, ys, {epochs:300}).then(() => {
  alert("完成訓練...");
  document.getElementById("output").innerText = "預測中...";
  // 預測資料  
  preds = model.predict(tf.tensor2d([26, 30], [2,1]));
  preds.array().then(array => {
    document.getElementById("output").innerText = array; 
  });
});
</script> 


<h4>當日氣溫預測業績的線性迴歸預測結果: </h4>
<p><span id="output">目前正在訓練中....</span></p>
</div>
</body>
</html>
```


##
```
<html>
<head>
<title>Ch15_1_2a.html</title>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-vis@latest"></script>
</head>
<body>
<script>
// 建立線性迴歸模型
const model = tf.sequential();
model.add(tf.layers.dense({units: 16, inputShape: [1]}));
model.add(tf.layers.dense({units: 1}));
model.compile({loss:"meanSquaredError", optimizer:"adam", metrics:["mse"]});
tfvis.show.modelSummary({name: "Model Summary"},model);
// 訓練資料
const xs = tf.tensor2d([29,28,34,31,25,29,32,
                        31,24,33,25,31,26,30], [14,1]);
const ys = tf.tensor2d([7.7,6.2,9.3,8.4,5.9,6.4,8.0,
                        7.5,5.8,9.1,5.1,7.3,6.5,8.4], [14,1]);
// 訓練模型
model.fit(xs, ys, {epochs: 120,shuffle: true, 
   callbacks: tfvis.show.fitCallbacks(
    {name: "Training Performance"},
    ["loss", "mse"],
    {height: 200, callbacks: ["onEpochEnd"]})
}).then(() => {
  alert("完成訓練...");
  document.getElementById("output").innerText = "預測中...";
  // 預測資料  
  preds = model.predict(tf.tensor2d([26, 30], [2,1]));
  preds.array().then(array => {
    document.getElementById("output").innerText = array; 
  });
});
</script> 
<h4>當日氣溫預測業績的線性迴歸預測結果: </h4>
<p><span id="output">目前正在訓練中....</span></p>
</div>
</body>
</html>
```


## 油耗預測佛型
```
使用線性迴歸神經網路以汽車馬力預測油耗
```
```
<!DOCTYPE html>
<html>

<head>
<meta charset="utf-8"/>
<title>Ch15_1_3.html</title>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-vis@latest"></script>
</head>


<body>
<script>
let data_path="https://storage.googleapis.com/tfjs-tutorials/carsData.json";
async function getData() {
  const carsDataReq = await fetch(data_path);
  const carsData = await carsDataReq.json();
  const cleaned = carsData.map(car => ({
    mpg: car.Miles_per_Gallon,
    horsepower: car.Horsepower,
  })).filter(car => (car.mpg != null && car.horsepower != null));
  return cleaned;
}  
async function visualization() {
  const values = (await getData()).map(d => ({
    x: d.horsepower,
    y: d.mpg,
  }));
  tfvis.render.scatterplot(
    { name: "Horsepower v MPG" },
    { values }, 
    { xLabel: "Horsepower", yLabel: "MPG",
      height: 300 }
  );
}
visualization();
</script>

</body>
</html>
```

# 分類
## 二元分類:XOR邏輯閘的神經網路模型
```
<html>
<head>
<title>Ch15_2_1.html</title>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-vis@latest"></script>
</head>
<body>
<script>
// 建立模型
function createModel() {
  let model = tf.sequential();
  model.add(tf.layers.dense({units:8, inputShape:2, activation: "tanh"}));
  model.add(tf.layers.dense({units:1, activation: "sigmoid"}));
  model.compile({optimizer: "sgd", loss: "binaryCrossentropy",
                 lr:0.1, metrics:["accuracy"]});
  tfvis.show.modelSummary({name: "Model Summary"},model);               
  return model;
}
const model = createModel();
// 訓練資料的張量
const xs = tf.tensor2d([[0,0],[0,1],[1,0],[1,1]]);
const ys = tf.tensor2d([[0],[1],[1],[0]]);
// 執行訓練
model.fit(xs, ys, {batchSize:1, epochs:3000, 
  callbacks: tfvis.show.fitCallbacks(
    { name: "Training Performance"},
    ["loss", "acc"],
    { yLabel: "loss/acc", height: 200,
      callbacks: ["onEpochEnd"]})
}).then(() => {
  alert("完成訓練...");
  document.getElementById("output").innerText = "預測中...";
  // 預測資料
  preds = model.predict(xs);
  preds.array().then(array => {
    document.getElementById("output").innerText = array; 
  });
});
</script> 
<h4>XOR 預測結果: </h4>
<p><span id="output">目前正在訓練中....</span></p>
</div>
</body>
</html>
```


## 多元分類:鳶尾花資料集的分類模型
```
鳶尾花資料集
https://fchart.github.io/test/iris.json
```

#### 鳶尾花資料集的資料視覺化
```
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>Ch15_2_2.html</title>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-vis@latest"></script>
</head>
<body>
<script>
let data_path="https://fchart.github.io/test/iris.json";  
async function getData() {
  const irisDataReq = await fetch(data_path); 
  const irisData = await irisDataReq.json(); 
  const filterData = irisData.map(flower => ({
    sLength: flower.sepalLength,
    sWidth: flower.sepalWidth,
    pLength: flower.petalLength,
    pWidth: flower.petalWidth
  }))
  .filter(flower => (flower.sLength != null && flower.sWidth != null 
               && flower.pLength != null && flower.pWidth != null));
  return filterData;
}
 
async function visualization() {
  const data = await getData();
  const sepals = data.map(d => ({ x: d.sLength, y: d.sWidth }));
  const petals = data.map(d => ({ x: d.pLength, y: d.pWidth }));
  tfvis.render.scatterplot(
    { name: "Sepal/Petal Length v Sepal/Petal Width" },
    { values: [sepals, petals], series: ["Sepal", "Petal"] },
    { xLabel: "Sepal/Petal Length", yLabel: "Sepal/Petal Width",
      height: 300 }
  );
}
visualization();
</script>
</body>
</html>
```


##
```



```


##
```



```


## 圖片識別與CNN卷積神經網路
```
MINIST手寫數字資料集
MNIST（Mixed National Institute of Standards and Technology）資料集是Yann Lecun’s提供的圖片資料庫
包含 60,000張手寫數字圖片（Handwritten Digit Image）的訓練資料集，和10,000張測試資料集。

MNIST資料集是成對的數字手寫圖片和對應的標籤資料0~9：
手寫數字圖片：尺寸28 x 28像素的灰階點陣圖。
標籤：手寫數字圖片對應實際的0~9數字。
```
#### CNN卷積神經網路
```
卷積層（Convolution Layers）
在卷積層是執行卷積運算，使用多個過濾器（Filters）或稱為卷積核（Kernels）掃瞄圖片來萃取出特徵
過濾器就是卷積層的權重（Weights）
```

```
池化層（Pooling Layers）
在池化層是執行池化運算，可以壓縮特徵圖來保留重要資訊，
目的是讓卷積神經網路專注於圖片中是否存在此特徵，而不是此特徵是位在哪裡？
```
#### MINIST手寫數字資料集的CNN卷積神經網路模型
```
<!DOCTYPE html>
<html>

<head>
<meta charset="utf-8"/>
<title>Ch15_3_2.html</title>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-vis@latest"></script>
<script src="data.js"></script>
</head>


<body>
<script>
async function showTestImgs(data) {
  const surface = tfvis.visor().surface(
      { name: "Test Data Examples", tab: "Test Data"});  

  const t_data = data.nextTestBatch(20);
  console.log("形狀: [" + t_data.xs.shape + "]");
  const size = t_data.xs.shape[0];

  for (let i = 0; i < size; i++) {
    const imgTensor = tf.tidy(() => {
      return t_data.xs
        .slice([i, 0], [1, t_data.xs.shape[1]])
        .reshape([28, 28, 1]);
    });
    
    const canvas = document.createElement("canvas");
    canvas.width = 28;
    canvas.height = 28;
    canvas.style = "margin: 4px;";
    await tf.browser.toPixels(imgTensor, canvas);
    surface.drawArea.appendChild(canvas);

    imgTensor.dispose();
  }
}

async function run() {  
  const data = new MnistData();
  await data.load();
  await showTestImgs(data);
}
run();
</script>
</body>

</html>
```

####
```
<!DOCTYPE html>
<html>

<head>
<meta charset="utf-8"/>
<title>Ch15_3_3.html</title>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-vis@latest"></script>
<script src="data.js"></script>
</head>


<body>
<script>
function createModel() {
  const model = tf.sequential();
  model.add(tf.layers.conv2d({
        inputShape: [28, 28, 1], kernelSize: 5, filters: 8,
        strides: 1, activation: "relu",
        kernelInitializer: "varianceScaling"}));
  model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2], strides: [2, 2]}));
  model.add(tf.layers.conv2d({
        kernelSize: 5, filters: 16, strides: 1,
        activation: "relu", kernelInitializer: "varianceScaling"}));
  model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2], strides: [2, 2]}));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 10, activation: "softmax",
        kernelInitializer: "varianceScaling",}));
  model.compile({loss: "categoricalCrossentropy", optimizer: "adam",
                 metrics:["accuracy"]});
  tfvis.show.modelSummary({name: "Model Summary"}, model);
  return model;
}

async function getData(){
  data = new MnistData();
  await data.load();
  return data;
}

function getTrainData(data, size) {
  return tf.tidy(() => {
    const d = data.nextTrainBatch(size);
      return {
        inputs: d.xs.reshape([size, 28, 28, 1]),
        labels: d.labels
      }
  });
}

function getTestData(data, size) {
  return tf.tidy(() => {
    const d = data.nextTestBatch(size);
      return {
        inputs: d.xs.reshape([size, 28, 28, 1]),
        labels: d.labels
      }
  });
}

async function trainModel(model, t_data,v_data) {
  const batchSize = 500;
  const epochs =10;
  return await model.fit(t_data.inputs, t_data.labels, {
    batchSize, epochs, shuffle: true,
    validationData: [v_data.inputs, v_data.labels],
    callbacks: tfvis.show.fitCallbacks(
      { name: "Training Performance" },
      ["loss", "val_loss", "acc", "val_acc"],
      { yLabel: "loss/acc", height: 200, 
        callbacks: ["onEpochEnd"] }
    )
  });
}

const classNames = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];

async function predictModel(model, data, size = 500) {
  const t_data = data.nextTestBatch(size);
  const t_xs = t_data.xs.reshape([size, 28, 28, 1]);
  const labels = t_data.labels.argMax(-1);
  const preds = model.predict(t_xs).argMax(-1);
  t_xs.dispose();
  const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
  const container = {name: "Accuracy", tab: "Evaluation"};
  tfvis.show.perClassAccuracy(container, classAccuracy, classNames);
  labels.dispose();
}

async function run(){
  const model = createModel();
  const data = await getData();
  const t_data = getTrainData(data, 5500);
  const v_data = getTestData(data, 1000);
  await trainModel(model, t_data, v_data);
  alert("完成訓練...");
  predictModel(model, data);
}
run();
</script>
</body>
</html>

```
##
```



```


##
```



```


##
```



```

