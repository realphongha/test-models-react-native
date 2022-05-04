import * as drawUtils from '../../utils/Draw';
import * as utils from '../../utils/Utils';
import * as Ort from 'onnxruntime-react-native';
import * as RNFS from 'react-native-fs';
import * as jpeg from 'jpeg-js';
import { Buffer } from 'buffer';


class UdpPose {
  constructor(modelPath, wInput, hInput, inputName, runTimes) {
    this.modelPath = modelPath;
    this.wInput = wInput;
    this.hInput = hInput;
    this.inputName = inputName;
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
    // normalizes and converts RGB to BGR:
    let normalizedImg = new Float32Array(this.wInput * this.hInput * 3);
    for (let i = 0; i < img.length; i += 3) {
      normalizedImg[i] = (img[i+2] - this.mean[2])/this.std[2];
      normalizedImg[i+1] = (img[i+1] - this.mean[1])/this.std[1];
      normalizedImg[i+2] = (img[i] - this.mean[0])/this.std[0];
    }
    // transposes:
    let wh = this.wInput*this.hInput;
    let w3 = this.wInput*3;
    const transposeImg = new Float32Array(this.wInput * this.hInput * 3);
    for (let i = 0; i < this.hInput; i += 1) {
        for (let j = 0; j < this.wInput; j += 1) {
            for (let k = 0; k < 3; k += 1) {
                transposeImg[k*wh+i*this.wInput+j] = normalizedImg[i*w3+j*3+k]
            }
        }
    } 
    // to tensor:
    const inputTensor = new Ort.Tensor('float32', transposeImg, 
      [1, 3, this.hInput, this.wInput]);
    return inputTensor;
  }

  postprocess(heatmaps, dims) {
    // dims = [1,17,64,48]
    let kp = 13; // only gets first 13 points
    let hmSize = dims[2]*dims[3];
    let newHeatmaps = new Float32Array(hmSize*kp);
    let j = 0;
    for (let i = 0; i < heatmaps.length; i++){
      if (i >= kp*hmSize){
        break;
      }
      newHeatmaps[j++] = heatmaps[i]; 
    }
    let points = [];
    let maxVals = [];
    for (let i = 0; i < kp; i++){
      let [maxIndex, maxVal] = utils.argmax(newHeatmaps.slice(i*hmSize, (i+1)*hmSize));
      maxVals.push((maxVal<0.0)?0.0:maxVal);
      if (maxVal < 0.0) {
        points.push([0, 0]);
      } else {
        let x = maxIndex%dims[3];
        let y = Math.floor(maxIndex/dims[3]);
        x = Math.round(x / dims[3] * this.wInput);
        y = Math.round(y / dims[2] * this.hInput);
        points.push([x, y]);
      }
    }
    return [points, maxVals];
  }

  async infer(img) {
    console.log("Session:", this.session);
    try {
      let preTimes = [];
      let modelTimes = [];
      let postTimes = [];
      var inputTensor;
      let dims;
      let heatmaps;
      let keypoints;
      let maxVals;
      for (let i = 0; i < this.runTimes; i++) {
        const t0 = performance.now();
        let objInput = {};
        inputTensor = this.preprocess(img);
        const t1 = performance.now();
        objInput[this.inputName] = inputTensor;
        let res = await this.session.run(objInput);
        heatmaps = res["output"]["data"];
        dims = res["output"]["dims"];
        const t2 = performance.now();
        [keypoints, maxVals] = this.postprocess(heatmaps, dims);
        const t3 = performance.now();
        preTimes.push(t1-t0);
        modelTimes.push(t2-t1);
        postTimes.push(t3-t2);
      }
      let latencyPre = utils.average(preTimes);
      let latencyModel = utils.average(modelTimes);
      let latencyPost = utils.average(postTimes);
      console.log("Keypoints:", keypoints);
      console.log("Scores:", maxVals);
      console.log("Latency (preprocess, model, postprocess):", 
        latencyPre, latencyModel, latencyPost);
      for (let i = 0; i < keypoints.length; i++){
        keypoints[i].push(maxVals[i]);
      }
      return keypoints;
    } catch (err) {
      console.log(err);
      return false;
    }
  }

  drawResult(img, keypoints) {
    let imgCopy = img.slice();
    let COLORS = {
      0: [255, 0, 0] , // "nose" - red
      1: [0, 255, 0] , // "left_eye" - green
      2: [0, 0, 255], // "right_eye" - blue
      3: [128, 128, 128], // "left_ear" - gray
      4: [255, 255, 0], // "right_ear" - yellow
      5: [0, 255, 255], // "left_shoulder" - cyan
      6: [255, 0, 255], // "right_shoulder" - pink
      7: [128, 0, 0], // "left_elbow" - dark red
      8: [128, 0, 128], // "right_elbow" - purple
      9: [255, 69, 0], // "left_wrist" - orange
      10: [139, 69, 19], // "right_wrist" - brown
      11: [0, 0, 0], // "left_hip" - black
      12: [192, 192, 192], // "right_hip" - silver
    }
    drawUtils.drawKeypointsColor(imgCopy, keypoints, this.wInput, this.hInput, 
      COLORS, 20);
    let frameData = new Buffer.alloc(this.wInput*this.hInput*4);
    for (let i = 0; i < frameData.length/4; i++){
      frameData[4*i] = imgCopy[3*i];
      frameData[4*i+1] = imgCopy[3*i+1];
      frameData[4*i+2] = imgCopy[3*i+2];
      frameData[4*i+3] = 255;
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
  UdpPose,
};