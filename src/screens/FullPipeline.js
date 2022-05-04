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
import { NanoDet } from '../components/infer_engine/NanoDet';
import { UdpPose } from '../components/infer_engine/UdpPose';
import { FCNet } from '../components/infer_engine/FCNet';
import ImageResizer from 'react-native-image-resizer';
import { LogBox } from 'react-native';

LogBox.ignoreLogs([
  'Non-serializable values were found in the navigation state',
]);

export default function FullPipeline({navigation, route}) {
  const [displayButton, setDisplayButton] = useState(false);
  const [testingTime, setTestingTime] = useState(0); // 0 - not tested yet
  // 1 - testing, 2 - tested
  const modelInput = route.params.modelInput;
  const inputShapeObjDet = [320, 320]; // w, h
  const inputShapePose = [192, 256]; // w, h
  const inputShapeAction = [13, 3]; // joints, channels
  const [actionResult, setActionResult] = useState(null);
  const [poseImgResult, setPoseImgResult] = useState(null);
  // const objdetPath = "weights/objdet/nanodet_plus_m_320.all.ort";
  const objdetPath = "weights/objdet/nanodet_plus_m_320_u8s8_excluded.basic.ort"
  const posePath = "weights/pose/pose_shufflenetv2_plus_pixel_shuffle.all.ort";
  const actionPath = "weights/action/fc_net.all.ort";
  const inputNameObjDet = "data";
  const inputNamePose = "images";
  const inputNameAction = "input";
  const numCls = 5;
  const regMax = 7;
  const strides = [8, 16, 32, 64];
  const iouThres = 0.2;
  const scoreThres = [0.4, 0.25, 0.25, 0.25, 0.25];
  const objdetEngine = new NanoDet(objdetPath, inputShapeObjDet[0], 
    inputShapeObjDet[1], inputNameObjDet, numCls, regMax, strides,
    iouThres, scoreThres, 10);
  const poseEngine = new UdpPose(posePath, inputShapePose[0], 
    inputShapePose[1], inputNamePose, 20);
  const actionEngine = new FCNet(actionPath, inputShapeAction[0], 
    inputShapeAction[1], inputShapePose[0], inputShapePose[1], 
    inputNameAction, 50);

  useEffect(() => {
    initEngine();
  }, []);

  const initEngine = async () => {
    await objdetEngine.initModel();
    await poseEngine.initModel();
    await actionEngine.initModel();
    setDisplayButton(true);
  }

  const testModel = async () => {
    try {
      if (!(objdetEngine.session && poseEngine.session && actionEngine.session)){
        await initEngine();
      }
      setTestingTime(1);
      setDisplayButton(false);
      let bboxes = await objdetEngine.infer(modelInput);
      const t1 = performance.now();
      let persons = [];
      for (let box of bboxes){
        if (box[5] === 0) {
          persons.push(box);
        }
      }
      if (persons.length === 0){
        console.log("There are no persons!")
        setTestingTime(0);
        setDisplayButton(true);
        return;
      }
      persons.sort((a, b) => b[4]-a[4]);
      console.log(persons);
      let person = persons[0]; // best person
      let [xmin, ymin, xmax, ymax, objScore, objCls] = person;
      xmin = Math.round(xmin);
      ymin = Math.round(ymin);
      xmax = Math.round(xmax);
      ymax = Math.round(ymax);
      // cuts person bbox:
      let newW = xmax-xmin;
      let newH = ymax-ymin;
      let newW4 = newW*4; 
      let w3 = inputShapeObjDet[0]*3;
      let personImg = new Buffer.alloc(newW*newH*4);
      for (let i = xmin; i <= xmax; i++){
        for (let j = ymin; j <= ymax; j++){
          personImg[(j-ymin)*newW4+(i-xmin)*4] = modelInput[j*w3+i*3];
          personImg[(j-ymin)*newW4+(i-xmin)*4+1] = modelInput[j*w3+i*3+1];
          personImg[(j-ymin)*newW4+(i-xmin)*4+2] = modelInput[j*w3+i*3+2];
          personImg[(j-ymin)*newW4+(i-xmin)*4+3] = 255;
        }
      }
      let rawImgData = {
        data: personImg,
        width: newW,
        height: newH,
      };
      let jpegImgData = jpeg.encode(rawImgData, 100);
      let jpegBase64 = jpegImgData["data"].toString('base64');
      jpegBase64 = `data:image/jpeg;base64,${jpegBase64}`;
      let newPath;
      await ImageResizer.createResizedImage(jpegBase64, 
        inputShapePose[0], inputShapePose[1], 'JPEG', 
        100, 0, undefined, false, {mode: 'stretch'})
        .then(response => {
          newPath = response.uri;
        })
        .catch(err => {
          console.log("Error resizing image:", err);
          return;
      });
      let imgBase64 = await RNFS.readFile(newPath, 'base64');
      let imgBuffer = Buffer.from(imgBase64, 'base64');
      let {width, height, data} = jpeg.decode(imgBuffer, {useTArray: true});
      // console.log(data);
      let poseInput = new Float32Array(width * height * 3);
      let offset = 0;
      for (let i = 0; i < poseInput.length; i += 3) {
        poseInput[i] = data[offset];
        poseInput[i + 1] = data[offset + 1];
        poseInput[i + 2] = data[offset + 2];
        offset += 4;
      }
      const t2 = performance.now();
      console.log("Preprocess objdet output to pose input (ms):", t2-t1);
      let keypoints = await poseEngine.infer(poseInput);
      console.log("Keypoints:", keypoints);
      let [cls, score, prob] = await actionEngine.infer(keypoints);
      let ACTION_NAME = {
        0: "Hand reach out",
        1: "Look down",
        2: "Look outside",
        3: "Sitting",
      }
      setActionResult(`Class: ${ACTION_NAME[cls]}, score: ${score}`);
      console.log(`Class: ${ACTION_NAME[cls]}, score: ${score}`);
      setTestingTime(2);
      setDisplayButton(true);
      // draws keypoints
      let jpegB64 = poseEngine.drawResult(poseInput, keypoints);
      setPoseImgResult(jpegB64);
    } catch (err) {
      console.log(err);
      setTestingTime(0);
      setDisplayButton(true);
    }
  }

  const showResult = () => {
    navigation.navigate("DisplayImage", {
      "imgBase64": poseImgResult,
      "text": actionResult,
    })
  }

  return (
    <View style={styles.container}>
      {(testingTime === 0) ?
        <Text style={styles.text}>
          Press Test model button...
        </Text> : ((testingTime === 1) ? <Text style={styles.text}>Testing...</Text> :
          <Text style={styles.text}>Done testing!</Text>)
      }
      {displayButton &&
        <Button
          style={styles.btn}
          onPress={() => testModel()}
          title="Test Full Pipeline"
        />}
      {((testingTime === 2) && poseImgResult && actionResult) &&
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