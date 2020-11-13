## AIBrowser瀏覽器的人工智慧
```
人工智慧的機器學習應用:使用TensorFlow.js在Web應用程式部署機器學習模型。

[1]使用Python在Google Colab雲端服務訓練MINIST模型的手寫數字辨識，
[2]然後在客戶端TensorFlow.js載入此模型來進行預測
```
# 推薦教科書
```
JavaScript 網頁設計與 TensorFlow.js 人工智慧應用教本
陳會安  碁峰資訊 2020-09-24
```

# 迴歸regression
## 線性迴歸
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


## 油耗預測模型
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
<!DOCTYPE html>
<html>

<head>
<meta charset="utf-8"/>
<title>Ch15_4.html</title>
<style>
#canvas {
  border:2px solid #000000;
  position: absolute;
  top: 80px;
  left: 50px;
}
#result {
  position: absolute;
  top: 400px;
  left: 50px;
}
</style>
<script src="jquery.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"></script>
<script>
var sigCanvas;
var context;

$(document).ready(function() {
  sigCanvas = document.getElementById("canvas");
  context = sigCanvas.getContext("2d");  
  context.strokeStyle = "#0000FF";
  context.lineJoin = "round";
  context.lineWidth = 20;
  $("#number").html("載入MNIST模型中...");
  tf.loadLayersModel('model/model.json').then(function(model) {
    window.model = model;
    $("#number").html("請使用滑鼠輸入數字...");
    $("#canvas").mousedown(function(mouseEvent) {
      let position = getPosition(mouseEvent, sigCanvas);
      context.moveTo(position.X, position.Y);
      context.beginPath();
      $(this).mousemove(function(mouseEvent) {
        drawLine(mouseEvent, sigCanvas, context);
      }).mouseup(function(mouseEvent) {
        finishDrawing(mouseEvent, sigCanvas, context);
      }).mouseout(function(mouseEvent) {
        finishDrawing(mouseEvent, sigCanvas, context);
      });
    });
  });
});

function getPosition(mouseEvent, sigCanvas) {
  let rect = sigCanvas.getBoundingClientRect();
  return {
    X: mouseEvent.clientX - rect.left,
    Y: mouseEvent.clientY - rect.top
  };
}

function drawLine(mouseEvent, sigCanvas, context) {
  let position = getPosition(mouseEvent, sigCanvas);
  context.lineTo(position.X, position.Y);
  context.stroke();
}

function finishDrawing(mouseEvent, sigCanvas, context) {
  drawLine(mouseEvent, sigCanvas, context);
  context.closePath();
  $(sigCanvas).unbind("mousemove")
    .unbind("mouseup")
    .unbind("mouseout");
}

function predictNum() {
  let img = new Image();
  img.onload = function() {
    context.drawImage(img, 0, 0, 28, 28);
    data = context.getImageData(0, 0, 28, 28).data;
    let input = [];
    for(let i = 0; i < data.length; i += 4) {
      input.push(data[i + 2] / 255);
    }
    let str = "";
    for (let i = 0; i < input.length; i++) {
      str += input[i];
      if ((i+1) % 28 == 0) 
        str += "<br/>";
    }
    $("#result").html(str);
    console.log(input);
    $("#number").html("");
    predictImg(input);
  };
  img.src = canvas.toDataURL('image/png');
}

var predictImg = function(input) {
  if (window.model) {
    window.model.predict([tf.tensor(input).reshape([1, 28, 28, 1])]).array()
    .then(function(scores){
      scores = scores[0];
      predicted = scores.indexOf(Math.max(...scores));
      $('#number').html("預測結果的數字: " + predicted);
    });
  } else {
    setTimeout(function(){predict(input)}, 50);
  }
}

function clearCanvas() {
  context.clearRect(0, 0, sigCanvas.width, sigCanvas.height);
  $("#number").html("請使用滑鼠輸入數字...");
}
</script>
</head>

<body>
<div>
  <canvas id="canvas" width="280px" height="280px"></canvas>
  <input type="button" value="清除數字" id="clearbutton" onclick="clearCanvas();">
  <input type="button" value="預測數字" id="predict" onclick="predictNum();">
  <div id="number">請使用滑鼠輸入數字...</div>
  <div id="result"></div>
</div>
</body>

</html>
```
