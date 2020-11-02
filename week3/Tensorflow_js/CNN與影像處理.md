#
```
CNN與影像處理
圖片識別與CNN卷積神經網路
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



##
```


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



##
```


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



##
```


```



## 即時姿勢偵測
```
使用PoseNet預訓練模型進行即時姿勢偵測
PoseNet預訓練模型可以在瀏覽器使用WebCam網路攝影機進行即時的人體姿勢偵測，支援單人或多人的姿勢偵測。

JavaScript程式使用<script>標籤載入PoseNet預訓練模型：
<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/posenet"></script>

只能偵測單人姿勢；多人會誤判
```
```
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Ch16_6.html</title>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/posenet"></script>
</head>
<body>
<video autoplay muted id="video" width="400" height="400"></video>
<canvas id="output" style="position:absolute;top:0;left:0;"></canvas>
<script>
const color = "aqua";  
const boundingBoxColor = "red";
const lineWidth = 2;

function toTuple({y, x}) {
  return [y, x];
}

function drawPoint(ctx, y, x, r, color) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2 * Math.PI);
  ctx.fillStyle = color;
  ctx.fill();
}   
    
function drawKeypoints(keypoints, minConfidence, ctx, scale=1) {
  for (let i = 0; i < keypoints.length; i++) {
    const keypoint = keypoints[i];
    if (keypoint.score < minConfidence) {
      continue;
    }
    const {y, x} = keypoint.position;
    drawPoint(ctx, y * scale, x * scale, 3, color);
  }
}
   
function drawSegment([ay, ax], [by, bx], color, scale, ctx) {
  ctx.beginPath();
  ctx.moveTo(ax * scale, ay * scale);
  ctx.lineTo(bx * scale, by * scale);
  ctx.lineWidth = lineWidth;
  ctx.strokeStyle = color;
  ctx.stroke();
}

function drawSkeleton(keypoints, minConfidence, ctx, scale=1) {
  const adjacentKeyPoints =
    posenet.getAdjacentKeyPoints(keypoints, minConfidence);

  adjacentKeyPoints.forEach((keypoints) => {
    drawSegment(
      toTuple(keypoints[0].position), toTuple(keypoints[1].position), color,
      scale, ctx);
  });
} 

async function setupWebcam() {
  video = document.getElementById("video");
  const stream = await navigator.mediaDevices.getUserMedia({
    "audio": false,
    "video": { facingMode: "user" },
  });
  video.srcObject = stream;
    
  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}
    
async function detect() {
  let minPoseConfidence = 0.1;
  let minPartConfidence = 0.5;
  const pose = await model.estimateSinglePose(video, {
    flipHorizontal: false
  });
  console.log(pose);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (pose.score >= minPoseConfidence) {
    drawKeypoints(pose.keypoints, minPartConfidence, ctx);
    drawSkeleton(pose.keypoints, minPartConfidence, ctx);
  }
  requestAnimationFrame(detect);
}

async function app() {
  await setupWebcam(); 
  video.play();
  videoWidth = video.videoWidth;
  videoHeight = video.videoHeight;
  video.width = videoWidth;
  video.height = videoHeight;
    
  canvas = document.getElementById("output");
  canvas.width = videoWidth;
  canvas.height = videoHeight;
  ctx = canvas.getContext('2d');
    
  model = await posenet.load({
    architecture: 'MobileNetV1',
    outputStride: 16,
    multiplier: 0.75
  });
  detect();
}
app();
</script>
</body>
</html>
```

```
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Ch16_6a.html</title>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/posenet"></script>
</head>
<body>
<video autoplay muted id="video" width="400" height="400"></video>
<canvas id="output" style="position:absolute;top:0;left:0;"></canvas>
<script>
const color = "aqua";  
const boundingBoxColor = "red";
const lineWidth = 2;

function toTuple({y, x}) {
  return [y, x];
}

function drawPoint(ctx, y, x, r, color) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2 * Math.PI);
  ctx.fillStyle = color;
  ctx.fill();
}   
    
function drawKeypoints(keypoints, minConfidence, ctx, scale=1) {
  for (let i = 0; i < keypoints.length; i++) {
    const keypoint = keypoints[i];
    if (keypoint.score < minConfidence) {
      continue;
    }
    const {y, x} = keypoint.position;
    drawPoint(ctx, y * scale, x * scale, 3, color);
  }
}
   
function drawSegment([ay, ax], [by, bx], color, scale, ctx) {
  ctx.beginPath();
  ctx.moveTo(ax * scale, ay * scale);
  ctx.lineTo(bx * scale, by * scale);
  ctx.lineWidth = lineWidth;
  ctx.strokeStyle = color;
  ctx.stroke();
}

function drawSkeleton(keypoints, minConfidence, ctx, scale=1) {
  const adjacentKeyPoints =
    posenet.getAdjacentKeyPoints(keypoints, minConfidence);

  adjacentKeyPoints.forEach((keypoints) => {
    drawSegment(
      toTuple(keypoints[0].position), toTuple(keypoints[1].position), color,
      scale, ctx);
  });
} 

async function setupWebcam() {
  video = document.getElementById("video");
  const stream = await navigator.mediaDevices.getUserMedia({
    "audio": false,
    "video": { facingMode: "user" },
  });
  video.srcObject = stream;
    
  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}
    
async function detect() {
  let minPoseConfidence = 0.15;
  let minPartConfidence = 0.1;
  let maxPoseDetections = 5;
  let nmsRadius = 30.0;

  const all_poses = await model.estimatePoses(video, {
    flipHorizontal: false,
    decodingMethod: "multi-person",
    maxDetections: maxPoseDetections,
    scoreThreshold: minPartConfidence,
    nmsRadius: nmsRadius
  });
  console.log(all_poses);    
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  all_poses.forEach(({score, keypoints}) => {
    if (score >= minPoseConfidence) {
      drawKeypoints(keypoints, minPartConfidence, ctx);
      drawSkeleton(keypoints, minPartConfidence, ctx);
    }
  });
  requestAnimationFrame(detect);
}

async function app() {
  await setupWebcam(); 
  video.play();
  videoWidth = video.videoWidth;
  videoHeight = video.videoHeight;
  video.width = videoWidth;
  video.height = videoHeight;
    
  canvas = document.getElementById("output");
  canvas.width = videoWidth;
  canvas.height = videoHeight;
  ctx = canvas.getContext('2d');
    
  model = await posenet.load({
    architecture: 'MobileNetV1',
    outputStride: 16,
    multiplier: 0.75
  });
  detect();
}
app();
</script>
</body>
</html>

```



