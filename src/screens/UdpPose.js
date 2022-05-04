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

export default function UdpPose({navigation, route}) {
  const [session, setSession] = useState(null);
  const [latencyPre, setLatencyPre] = useState(null);
  const [latencyModel, setLatencyModel] = useState(null);
  const [latencyPost, setLatencyPost] = useState(null);
  const [testingTime, setTestingTime] = useState(0); // 0 - not tested yet
  // 1 - testing, 2 - tested
  const [result, setResult] = useState(null);
  const modelInput = route.params.modelInput;
  const wInput = 192;
  const hInput = 256;
  const testTime = 10;
  const path = "weights/pose/pose_shufflenetv2_plus_pixel_shuffle.all.ort";
  // const path = "weights/pose/pose_shufflenetv2_plus_pixel_shuffle_u8s8.basic.ort";
  // const std = [0.229, 0.224, 0.225];
  // const mean = [0.485, 0.456, 0.406];
  const std = [57.375, 57.12, 58.395];
  const mean = [103.53, 116.28, 123.675];
  const inputName = "images";

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
    // normalizes and converts RGB to BGR:
    let normalizedImg = new Float32Array(wInput * hInput * 3);
    for (let i = 0; i < img.length; i += 3) {
      normalizedImg[i] = (img[i+2] - mean[2])/std[2];
      normalizedImg[i+1] = (img[i+1] - mean[1])/std[1];
      normalizedImg[i+2] = (img[i] - mean[0])/std[0];
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

  const postprocess = (heatmaps, dims) => {
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
        x = Math.round(x / dims[3] * wInput);
        y = Math.round(y / dims[2] * hInput);
        points.push([x, y]);
      }
    }
    return [points, maxVals];
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
      let heatmaps;
      let keypoints;
      let maxVals;
      for (let i = 0; i < testTime; i++) {
        const t0 = performance.now();
        let objInput = {};
        inputTensor = preprocess(img);
        const t1 = performance.now();
        objInput[inputName] = inputTensor;
        let res = await session.run(objInput);
        heatmaps = res["output"]["data"];
        dims = res["output"]["dims"];
        // console.log(Object.keys(res["output"]));
        const t2 = performance.now();
        [keypoints, maxVals] = postprocess(heatmaps, dims);
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
      setResult(keypoints);
      console.log(keypoints);
      console.log(maxVals);
    } catch (err) {
      console.log(err);
    }
  }

  const showResult = () => {
    let imgCopy = modelInput.slice();
    drawUtils.drawKeypoints(imgCopy, result, wInput, hInput, [255, 0, 255], 10);
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