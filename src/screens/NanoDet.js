import React, { useState, useEffect, useRef } from 'react';
import {
  Text,
  View,
  TouchableWithoutFeedback,
  StyleSheet,
  Button,
  Image,
  Platform
} from 'react-native';
import * as constants from '../utils/Constant';
import * as drawUtils from '../utils/Draw';
import * as utils from '../utils/Utils';
import * as Ort from 'onnxruntime-react-native';
import * as RNFS from 'react-native-fs';
import * as jpeg from 'jpeg-js';
import { Buffer } from 'buffer';
import { LogBox } from 'react-native';

LogBox.ignoreLogs([
  'Non-serializable values were found in the navigation state',
]);

export default function NanoDet({navigation, route}) {
  const [session, setSession] = useState(null);
  const [latencyPre, setLatencyPre] = useState(null);
  const [latencyModel, setLatencyModel] = useState(null);
  const [latencyPost, setLatencyPost] = useState(null);
  const [testingTime, setTestingTime] = useState(0); // 0 - not tested yet
  // 1 - testing, 2 - tested
  const [result, setResult] = useState(null);
  const modelInput = route.params.modelInput;
  const wInput = 320;
  const hInput = 320;
  const testTime = 10;
  // const path = "weights/objdet/nanodet_plus_m_320.all.ort";
  const path = "weights/objdet/nanodet_plus_m_320_coco_exam_iith_dmu_gen_data.all.ort"
  const std = [57.375, 57.12, 58.395];
  const mean = [103.53, 116.28, 123.675];
  const inputName = "data";
  const numCls = 5;
  const regMax = 7;
  const strides = [8, 16, 32, 64];
  const iouThres = 0.2;
  const scoreThres = [0.4, 0.25, 0.25, 0.25, 0.25];

  useEffect(() => {
    initModel();
  }, []);

  const initModel = async () => {
    let copyDstPath = RNFS.TemporaryDirectoryPath + "/" +
      path.replace(new RegExp("/", 'g'), "_");
    let absPath = null;
    if (Platform.OS == "android") {
      try {
        await RNFS.copyFileAssets(path, copyDstPath);
        absPath = "file://" + copyDstPath;
        console.log("Abs path:", absPath);
      } catch (err) {
        console.log(err);
        return (
            <View style={styles.container}>
              <Text>Cannot load model!</Text>
            </View>
          )
      }
    } else { // ios
      return (
        <View style={styles.container}>
          <Text>Platform not supported!</Text>
        </View>
      )
    }
    if (absPath) {
      try {
        let sess = await Ort.InferenceSession.create(absPath);
        setSession(sess);
      } catch (err) {
        console.log("Err:", err);
        return (
            <View style={styles.container}>
              <Text>Cannot load model!</Text>
            </View>
          )
      }
    } else {
      console.log("Cannot load model!");
      return (
        <View style={styles.container}>
          <Text>Cannot load model!</Text>
        </View>
      )
    }
  }

  const preprocess = (img) => {
    // normalizes:
    let normalizedImg = new Float32Array(wInput * hInput * 3);
    for (let i = 0; i < img.length; i += 3) {
      normalizedImg[i] = (img[i] - mean[0])/std[0];
      normalizedImg[i+1] = (img[i+1] - mean[1])/std[1];
      normalizedImg[i+2] = (img[i+2] - mean[2])/std[2];
    }
    // transposes:
    let wh = wInput*hInput;
    let w3 = wInput*3;
    const transposeImg = new Float32Array(wInput * hInput * 3);
    for (let i = 0; i < hInput; i += 1) {
        for (let j = 0; j < wInput; j += 1) {
            for (let k = 0; k < 3; k += 1) {
                transposeImg[k*wh+i*wInput+j] = normalizedImg[i*w3+j*3+k]
            }
        }
    } 
    // to tensor:
    const inputTensor = new Ort.Tensor('float32', transposeImg, 
      [1, 3, hInput, wInput]);
    return inputTensor;
  }

  const distance2bbox = (dflPred, x, y, stride) => {
    let ctX = x*stride;
    let ctY = y*stride;
    let disPred = [0, 0, 0, 0];
    let len = regMax + 1;
    for (let i = 0; i < 4; i++){
      let dis = 0;
      let disAfterSm = [];
      let idx = i * len;
      let alpha = Math.max(...dflPred.slice(idx, idx+len))
      for (let j = 0; j < len; j++){
        disAfterSm.push(Math.exp(dflPred[idx+j]-alpha));
      }
      let maxDisAfterSm = disAfterSm.reduce((a, b) => a + b, 0);
      for (let j = 0; j < len; j++){
        disAfterSm[j] /= maxDisAfterSm;
      } 
      for (let j = 0; j < len; j++){
        dis += j * disAfterSm[j];
      }
      dis *= stride;
      disPred[i] = dis;
    }
    let bbox = [];
    bbox.push(Math.max(ctX-disPred[0], 0));
    bbox.push(Math.max(ctY-disPred[1], 0));
    bbox.push(Math.min(ctX+disPred[2], wInput));
    bbox.push(Math.min(ctY+disPred[3], hInput));
    return bbox;
  }

  const sortFunc = (a, b) => {
    return b[4]-a[4];
  }

  const iouCalc = (box1, box2) => {
    let box1Area = (box1[2]-box1[0])*(box1[3]-box1[1]);
    let box2Area = (box2[2]-box2[0])*(box2[3]-box2[1]);
    let leftUp = [(box1[0]>box2[0])?box1[0]:box2[0], (box1[1]>box2[1])?box1[1]:box2[1]];
    let rightDown = [(box1[2]<box2[2])?box1[2]:box2[2], (box1[3]<box2[3])?box1[3]:box2[3]];
    let intersect = [rightDown[0]-leftUp[0], rightDown[1]-leftUp[1]];
    intersect[0] = (intersect[0]<0.0)?0.0:intersect[0];
    intersect[1] = (intersect[1]<0.0)?0.0:intersect[1];
    let intersectArea = intersect[0]*intersect[1];
    // return 1.0*intersectArea/(box1Area+box2Area-intersectArea);
    return 1.0*intersectArea/((box1Area<box2Area)?box1Area:box2Area);
  }

  const multiclassNms = (bboxes) => {
    if (bboxes.length == 0){
      return [];
    }
    bboxes.sort(sortFunc);
    let returnBoxes = [];
    let boxesDict = {};
    bboxes.forEach(box => {
      let cls = box[5];
      if (cls in boxesDict){
        boxesDict[cls].push(box);
      } else {
        boxesDict[cls] = [box];
      }
    });

    for (let cls in boxesDict){
      let boxs = boxesDict[cls];
      if (boxs.length === 1) {
        returnBoxes.push(boxs[0]);
      } else {
        while (boxs.length){
          let bestBox = boxs.splice(0, 1)[0];
          returnBoxes.push(bestBox);
          for (let i = 0; i < boxs.length; i++){
            if (iouCalc(bestBox, boxs[i]) > iouThres){
              boxs.splice(i, 1);
              i -= 1;
            }
          }
        }
      }
    }
    return returnBoxes;
  }

  const postprocess = (output, dims) => {
    let results = [];
    let clsPreds = [];
    let disPreds = [];
    let disFeatures = dims[2]-numCls;
    for (let i = 0; i < dims[1]; i++) {
      for (let j = 0; j < dims[2]; j++){
        if (j < numCls){
          clsPreds.push(output[i*dims[2]+j]);
        } else {
          disPreds.push(output[i*dims[2]+j]);
        }
      }
    }
    // console.log(clsPreds);

    let center_priors = [];
    strides.forEach(stride => {
      let featW = Math.ceil(wInput/stride);
      let featH = Math.ceil(hInput/stride);
      for (let y = 0; y < featH; y++){
        for (let x = 0; x < featW; x++){
          center_priors.push(x);
          center_priors.push(y);
          center_priors.push(stride);
        }
      }
    })

    for (let i = 0; i < center_priors.length/3; i++) {
      let x = center_priors[3*i];
      let y = center_priors[3*i+1];
      let stride = center_priors[3*i+2];
      let [maxCls, score] = utils.argmax(clsPreds.slice(i*numCls, (i+1)*numCls));
      if (score > scoreThres[maxCls]){
        // console.log(score, maxCls);
        let bbox = distance2bbox(disPreds.slice(i*disFeatures, (i+1)*disFeatures), x, y, stride);
        bbox.push(score);
        bbox.push(maxCls);
        results.push(bbox);
      }
    }

    return multiclassNms(results);
  }

  const testModel = async (img) => {
    console.log("Session:", session);
    try {
      let preTimes = new Array();
      let modelTimes = new Array();
      let postTimes = new Array();
      setTestingTime(1);
      var inputTensor;
      let dims;
      let result;
      let boxes;
      for (let i = 0; i < testTime; i++) {
        const t0 = performance.now();
        let objInput = {};
        inputTensor = preprocess(img);
        const t1 = performance.now();
        objInput[inputName] = inputTensor;
        let res = await session.run(objInput);
        result = res["output"]["data"];
        dims = res["output"]["dims"];
        const t2 = performance.now();
        boxes = postprocess(result, dims);
        const t3 = performance.now();
        preTimes.push(t1-t0);
        modelTimes.push(t2-t1);
        postTimes.push(t3-t2);
        setLatencyPre(utils.average(preTimes));
        setLatencyModel(utils.average(modelTimes));
        setLatencyPost(utils.average(postTimes));
        setTestingTime(2);
        console.log("Preprocess latency:", t1-t0);
        console.log("Model latency:", t2-t1);
        console.log("Postprocess latency:", t3-t2);
      }
      console.log("Final results:", boxes);
      setResult(boxes);
    } catch (err) {
      console.log(err);
    }
  }

  const showResult = () => {
    const COLORS = {
      0: [255, 0, 0], // red - person
      1: [0, 255, 0], // green - laptop
      2: [0, 0, 255], // blue - mouse
      3: [255, 0, 255], // pink - keyboard
      4: [255, 255, 0], // yellow - cell phone
    }
    let imgCopy = modelInput.slice();
    for (let box of result){
      let [xmin, ymin, xmax, ymax, score, cls] = box;
      console.log(xmin, ymin, xmax, ymax, score, cls);
      drawUtils.drawBox(imgCopy, xmin, ymin, xmax, ymax, wInput, hInput, 
        COLORS[cls], 10);
    }
    let frameData = new Buffer.alloc(wInput*hInput*4);
    for (let i = 0; i < frameData.length/4; i++){
      frameData[4*i] = imgCopy[3*i];
      frameData[4*i+1] = imgCopy[3*i+1];
      frameData[4*i+2] = imgCopy[3*i+2];
      frameData[4*i+3] = 255;
    }
    let rawImgData = {
      data: frameData,
      width: wInput,
      height: hInput,
    };
    let jpegImgData = jpeg.encode(rawImgData, 50);
    // console.log(jpegImgData);
    let jpegBase64 = jpegImgData["data"].toString('base64');
    navigation.navigate("DisplayImage", {
      "imgBase64": jpegBase64,
    });
  }

  return (
    <View style={styles.container}>
      <Text style={styles.bigText}>Model: {path}</Text>
      {(testingTime === 0) ?
        <Text style={styles.text}>
          Press Test model button...
        </Text> : ((testingTime === 1) ? <Text style={styles.text}>Testing...</Text> :
          <View>
            <Text style={styles.text}>
              Preprocess FPS: {(1 / latencyPre * 1000).toFixed(2)}, Latency: {latencyPre.toFixed(2)}(ms)
            </Text>
            <Text style={styles.text}>
              Model FPS: {(1 / latencyModel * 1000).toFixed(2)}, Latency: {latencyModel.toFixed(2)}(ms)
            </Text>
            <Text style={styles.text}>
              Postprocess FPS: {(1 / latencyPost * 1000).toFixed(2)}, Latency: {latencyPost.toFixed(2)}(ms)
            </Text>
          </View>)
      }
      {session &&
        <Button
          style={styles.btn}
          onPress={() => testModel(modelInput)}
          title="Test model speed"
        />}
      {result &&
        <Button
          style={styles.btn}
          onPress={() => showResult()}
          title="Show result"
        />}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignSelf: 'stretch',
    backgroundColor: constants.gray,
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center'
  },
  btn: {
    paddingTop: 70,
  },
  text: {
    alignSelf: 'center',
    color: '#fff'
  },
  bigText: {
    alignSelf: 'center',
    color: '#fff',
    fontSize: 20,
  }
});