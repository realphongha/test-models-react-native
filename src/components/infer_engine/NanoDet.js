import * as drawUtils from '../../utils/Draw';
import * as utils from '../../utils/Utils';
import * as Ort from 'onnxruntime-react-native';
import * as RNFS from 'react-native-fs';
import * as jpeg from 'jpeg-js';
import { Buffer } from 'buffer';


class NanoDet {
  constructor(modelPath, wInput, hInput, inputName, numCls, regMax, strides,
    iouThres, scoreThres, runTimes) {
    this.modelPath = modelPath;
    this.wInput = wInput;
    this.hInput = hInput;
    this.inputName = inputName;
    this.numCls = numCls;
    this.regMax = regMax;
    this.strides = strides;
    this.iouThres = iouThres;
    this.scoreThres = scoreThres;
    this.std = [57.375, 57.12, 58.395];
    this.mean = [103.53, 116.28, 123.675];
    this.runTimes = runTimes;
    this.session = null;
  }

  async initModel() {
    let copyDstPath = RNFS.TemporaryDirectoryPath + "/" +
      this.modelPath.replace(new RegExp("/", 'g'), "_");
    let absPath = null;
    if (Platform.OS == "android") {
      try {
        await RNFS.copyFileAssets(this.modelPath, copyDstPath);
        absPath = "file://" + copyDstPath;
        console.log("Abs path:", absPath);
      } catch (err) {
        console.log(err);
        return false;
      }
    } else { // ios
      console.log("Platform not supported yet!")
      return false;
    }
    if (absPath) {
      try {
        let sess = await Ort.InferenceSession.create(absPath);
        this.session = sess;
      } catch (err) {
        console.log("Err:", err);
        return false;
      }
    } else {
      console.log("Cannot load model!");
      return false;
    }
    return true;
  }

  preprocess(img) {
    // normalizes:
    let normalizedImg = new Float32Array(this.wInput * this.hInput * 3);
    for (let i = 0; i < img.length; i += 3) {
      normalizedImg[i] = (img[i] - this.mean[0]) / this.std[0];
      normalizedImg[i + 1] = (img[i + 1] - this.mean[1]) / this.std[1];
      normalizedImg[i + 2] = (img[i + 2] - this.mean[2]) / this.std[2];
    }
    // transposes:
    let wh = this.wInput * this.hInput;
    let w3 = this.wInput * 3;
    const transposeImg = new Float32Array(this.wInput * this.hInput * 3);
    for (let i = 0; i < this.hInput; i += 1) {
      for (let j = 0; j < this.wInput; j += 1) {
        for (let k = 0; k < 3; k += 1) {
          transposeImg[k * wh + i * this.wInput + j] = normalizedImg[i * w3 + j * 3 + k]
        }
      }
    }
    // to tensor:
    const inputTensor = new Ort.Tensor('float32', transposeImg,
      [1, 3, this.hInput, this.wInput]);
    return inputTensor;
  }

  distance2bbox(dflPred, x, y, stride) {
    let ctX = x * stride;
    let ctY = y * stride;
    let disPred = [0, 0, 0, 0];
    let len = this.regMax + 1;
    for (let i = 0; i < 4; i++) {
      let dis = 0;
      let disAfterSm = [];
      let idx = i * len;
      let alpha = Math.max(...dflPred.slice(idx, idx + len))
      for (let j = 0; j < len; j++) {
        disAfterSm.push(Math.exp(dflPred[idx + j] - alpha));
      }
      let maxDisAfterSm = disAfterSm.reduce((a, b) => a + b, 0);
      for (let j = 0; j < len; j++) {
        disAfterSm[j] /= maxDisAfterSm;
      }
      for (let j = 0; j < len; j++) {
        dis += j * disAfterSm[j];
      }
      dis *= stride;
      disPred[i] = dis;
    }
    let bbox = [];
    bbox.push(Math.max(ctX - disPred[0], 0));
    bbox.push(Math.max(ctY - disPred[1], 0));
    bbox.push(Math.min(ctX + disPred[2], this.wInput-1));
    bbox.push(Math.min(ctY + disPred[3], this.hInput-1));
    return bbox;
  }

  iouCalc(box1, box2) {
    let box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    let box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1]);
    let leftUp = [(box1[0] > box2[0]) ? box1[0] : box2[0], (box1[1] > box2[1]) ? box1[1] : box2[1]];
    let rightDown = [(box1[2] < box2[2]) ? box1[2] : box2[2], (box1[3] < box2[3]) ? box1[3] : box2[3]];
    let intersect = [rightDown[0] - leftUp[0], rightDown[1] - leftUp[1]];
    intersect[0] = (intersect[0] < 0.0) ? 0.0 : intersect[0];
    intersect[1] = (intersect[1] < 0.0) ? 0.0 : intersect[1];
    let intersectArea = intersect[0] * intersect[1];
    // return 1.0*intersectArea/(box1Area+box2Area-intersectArea);
    return 1.0 * intersectArea / ((box1Area < box2Area) ? box1Area : box2Area);
  }

  multiclassNms(bboxes) {
    if (bboxes.length == 0) {
      return [];
    }
    bboxes.sort((a, b) => {
      return b[4] - a[4];
    });
    let returnBoxes = [];
    let boxesDict = {};
    bboxes.forEach(box => {
      let cls = box[5];
      if (cls in boxesDict) {
        boxesDict[cls].push(box);
      } else {
        boxesDict[cls] = [box];
      }
    });

    for (let cls in boxesDict) {
      let boxs = boxesDict[cls];
      if (boxs.length === 1) {
        returnBoxes.push(boxs[0]);
      } else {
        while (boxs.length) {
          let bestBox = boxs.splice(0, 1)[0];
          returnBoxes.push(bestBox);
          for (let i = 0; i < boxs.length; i++) {
            if (this.iouCalc(bestBox, boxs[i]) > this.iouThres) {
              boxs.splice(i, 1);
              i -= 1;
            }
          }
        }
      }
    }
    return returnBoxes;
  }

  postprocess(output, dims) {
    let results = [];
    let clsPreds = [];
    let disPreds = [];
    let disFeatures = dims[2] - this.numCls;
    for (let i = 0; i < dims[1]; i++) {
      for (let j = 0; j < dims[2]; j++) {
        if (j < this.numCls) {
          clsPreds.push(output[i * dims[2] + j]);
        } else {
          disPreds.push(output[i * dims[2] + j]);
        }
      }
    }
    // console.log(clsPreds);

    let center_priors = [];
    this.strides.forEach(stride => {
      let featW = Math.ceil(this.wInput / stride);
      let featH = Math.ceil(this.hInput / stride);
      for (let y = 0; y < featH; y++) {
        for (let x = 0; x < featW; x++) {
          center_priors.push(x);
          center_priors.push(y);
          center_priors.push(stride);
        }
      }
    })

    for (let i = 0; i < center_priors.length / 3; i++) {
      let x = center_priors[3 * i];
      let y = center_priors[3 * i + 1];
      let stride = center_priors[3 * i + 2];
      let [maxCls, score] = utils.argmax(clsPreds.slice(i * this.numCls, 
        (i + 1) * this.numCls));
      if (score > this.scoreThres[maxCls]) {
        // console.log(score, maxCls);
        let bbox = this.distance2bbox(disPreds.slice(i * disFeatures, (i + 1) * disFeatures), x, y, stride);
        bbox.push(score);
        bbox.push(maxCls);
        results.push(bbox);
      }
    }

    return this.multiclassNms(results);
  }

  async infer(img) {
    console.log("Session:", this.session);
    try {
      let preTimes = [];
      let modelTimes = [];
      let postTimes = [];
      var inputTensor;
      let dims;
      let result;
      let boxes;
      for (let i = 0; i < this.runTimes; i++) {
        const t0 = performance.now();
        let objInput = {};
        inputTensor = this.preprocess(img);
        const t1 = performance.now();
        objInput[this.inputName] = inputTensor;
        let res = await this.session.run(objInput);
        result = res["output"]["data"];
        dims = res["output"]["dims"];
        const t2 = performance.now();
        boxes = this.postprocess(result, dims);
        const t3 = performance.now();
        preTimes.push(t1 - t0);
        modelTimes.push(t2 - t1);
        postTimes.push(t3 - t2);
      }
      let latencyPre = utils.average(preTimes);
      let latencyModel = utils.average(modelTimes);
      let latencyPost = utils.average(postTimes);
      console.log("Final results:", boxes);
      console.log("Latency (preprocess, model, postprocess):", 
        latencyPre, latencyModel, latencyPost)
      return boxes;
    } catch (err) {
      console.log(err);
      return false;
    }
  }

  drawResult(img, bboxes) {
    const COLORS = {
      0: [255, 0, 0], // red - person
      1: [0, 255, 0], // green - laptop
      2: [0, 0, 255], // blue - mouse
      3: [255, 0, 255], // pink - keyboard
      4: [255, 255, 0], // yellow - cell phone
    }
    let imgCopy = img.slice();
    for (let box of bboxes) {
      let [xmin, ymin, xmax, ymax, score, cls] = box;
      console.log(xmin, ymin, xmax, ymax, score, cls);
      drawUtils.drawBox(imgCopy, xmin, ymin, xmax, ymax, 
        this.wInput, this.hInput, COLORS[cls], 10);
    }
    let frameData = new Buffer.alloc(this.wInput * this.hInput * 4);
    for (let i = 0; i < frameData.length / 4; i++) {
      frameData[4 * i] = imgCopy[3 * i];
      frameData[4 * i + 1] = imgCopy[3 * i + 1];
      frameData[4 * i + 2] = imgCopy[3 * i + 2];
      frameData[4 * i + 3] = 255;
    }
    let rawImgData = {
      data: frameData,
      width: this.wInput,
      height: this.hInput,
    };
    let jpegImgData = jpeg.encode(rawImgData, 50);
    // console.log(jpegImgData);
    let jpegBase64 = jpegImgData["data"].toString('base64');
    return jpegBase64;
  }
}

module.exports = {
  NanoDet,
};