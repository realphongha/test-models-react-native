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
import * as Ort from 'onnxruntime-react-native';
import * as RNFS from 'react-native-fs';

export default function ModelSpeedTestAbs(props) {
  const [session, setSession] = useState(null);
  const wInput = props.w;
  const hInput = props.h;
  const testTime = props.testTime;
  const [latency, setLatency] = useState(0);
  const [testingTime, setTestingTime] = useState(0); // 0 - not tested yet
  // 1 - testing, 2 - tested
  const path = props.modelPath;

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
      }
    } else {
      return (
        <View style={styles.container}>
          <Text>Cannot load model!</Text>
        </View>
      )
    }
  }

  const average = arr => arr.reduce((p, c) => p + c, 0) / arr.length;

  const testModel = async () => {
    console.log("Session:", session);
    let testTensor = new Ort.Tensor('float32',
      new Float32Array(wInput * hInput * 3), [1, 3, hInput, wInput]);
    // console.log(testTensor);
    try {
      // console.log(testTensor);
      let times = new Array();
      setTestingTime(1);
      for (let i = 0; i < testTime; i++) {
        const t0 = performance.now();
        let objInput = {};
        objInput[props.inputName] = testTensor;
        let res = await session.run(objInput);
        // console.log(res);
        const t1 = performance.now();
        let time = t1 - t0;
        console.log(time);
        times.push(time);
      }
      setTestingTime(2);
      setLatency(average(times));
    } catch (err) {
      console.log(err);
    }
  }

  return (
    <View style={styles.container}>
      <Text style={styles.bigText}>Model: {path}</Text>
      {(testingTime === 0) ?
        <Text style={styles.text}>
          Press Test model button...
        </Text> : ((testingTime === 1) ? <Text style={styles.text}>Testing...</Text> :
          <Text style={styles.text}>
            FPS: {(1 / latency * 1000).toFixed(2)}, Latency: {latency.toFixed(2)}(ms)
          </Text>)
      }
      {session &&
        <Button
          style={styles.btn}
          onPress={() => testModel()}
          title="Test model speed"
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
