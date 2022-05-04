import * as utils from '../../utils/Utils';
import * as Ort from 'onnxruntime-react-native';
import * as RNFS from 'react-native-fs';


class FCNet {
  constructor(modelPath, joints, channels, w, h, inputName, runTimes) {
    this.modelPath = modelPath;
    this.joints = joints;
    this.channels = channels;
    this.w = w;
    this.h = h;
    this.inputName = inputName;
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

  preprocess(rawKeypoints) {
    let keypoints = [];
    for (let point of rawKeypoints){
      keypoints.push(point.slice());
    }
    // normalizes:
    for (let point of keypoints){
      point[0] /= this.w;
      point[1] /= this.h;
    }
    let min0 = 9999999;
    let max0 = -9999999;
    let min1 = 9999999;
    let max1 = -9999999;
    for (let point of keypoints){
      if (point[0] < min0) min0 = point[0];
      if (point[0] > max0) max0 = point[0];
      if (point[1] < min1) min1 = point[1];
      if (point[1] > max1) max1 = point[1];
    }
    let normalizedKeypoints = new Float32Array(this.joints*this.channels);
    let i = 0;
    for (let point of keypoints){
      normalizedKeypoints[i++] = (point[0]-min0)/(max0-min0);
      normalizedKeypoints[i++] = (point[1]-min1)/(max1-min1);
      normalizedKeypoints[i++] = point[2];
    }
    const inputTensor = new Ort.Tensor('float32', normalizedKeypoints, 
      [1, this.joints, this.channels]);
    return inputTensor;
  }

  postprocess(output) {
    let prob = utils.softmax(output);
    let result = utils.argmax(prob); // [cls, score]
    result.push(prob);
    return result;
  }

  async infer(keypoints) {
    console.log("Session:", this.session);
    try {
      let preTimes = [];
      let modelTimes = [];
      let postTimes = [];
      var inputTensor;
      let output;
      let cls;
      let score;
      let prob;
      for (let i = 0; i < this.runTimes; i++) {
        const t0 = performance.now();
        let objInput = {};
        inputTensor = this.preprocess(keypoints);
        // console.log(inputTensor);
        const t1 = performance.now();
        objInput[this.inputName] = inputTensor;
        let res = await this.session.run(objInput);
        output = res["output"]["data"];
        const t2 = performance.now();
        [cls, score, prob] = this.postprocess(output);
        const t3 = performance.now();
        preTimes.push(t1-t0);
        modelTimes.push(t2-t1);
        postTimes.push(t3-t2);
      }
      let latencyPre = utils.average(preTimes);
      let latencyModel = utils.average(modelTimes);
      let latencyPost = utils.average(postTimes);
      console.log("Class:", cls);
      console.log("Score:", score);
      console.log("Prob:", prob)
      console.log("Latency (preprocess, model, postprocess):", 
        latencyPre, latencyModel, latencyPost)
      return [cls, score];
    } catch (err) {
      console.log(err);
      return false;
    }
  }
}

module.exports = {
  FCNet,
};