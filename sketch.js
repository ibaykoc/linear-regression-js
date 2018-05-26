let x_Vals = [];
let y_Vals = [];

function setup() {
  createCanvas(400, 400);
  m = tf.variable(tf.scalar(random(1)));
  b = tf.variable(tf.scalar(random(1)));
}

const learningRate = 0.1;
const optimizer = tf.train.sgd(learningRate);

function predict(xs) {
  const tfxs = tf.tensor1d(xs);
  const y_pred = tfxs.mul(m).add(b);
  return y_pred;
}

function loss(pred, labels) {
  return pred.sub(labels).square().mean();
}

function mousePressed() {
  let x = map(mouseX, 0, width, 0, 1);
  let y = map(mouseY, 0, height, 1, 0);
  x_Vals.push(x);
  y_Vals.push(y);
}

function draw() {

  background(0);

  if(x_Vals.length > 0) {
    
    tf.tidy(() => {
      const y_tf = tf.tensor1d(y_Vals);
      optimizer.minimize(() => loss(predict(x_Vals), y_tf));
    })

    stroke(255);
    strokeWeight(10);

    for (let i = 0; i < x_Vals.length; i++) {
      const x = map(x_Vals[i], 0, 1, 0, width);
      const y = map(y_Vals[i], 0, 1,height, 0);
      point(x, y);
    }
      let yPred = tf.tidy(() => predict([0, 1]));

      let x1 = 0;
      let x2 = width;

      let yData = yPred.dataSync();
      let y1 = map(yData[0], 0, 1, height, 0);
      let y2 = map(yData[1], 0, 1, height, 0);
      strokeWeight(4);

      line(x1, y1, x2, y2);

      yPred.dispose()
  }
  // noLoop()
  print(tf.memory().numTensors);
}

//file:///C:/dev/jsProj/LinearRegressionTF/index.html