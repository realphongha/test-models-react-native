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

export default function FCNetMO(props) {
  const [session, setSession] = useState(null);
  const joints = 13;
  const channels = 3;
  const testTime = 50;
  const [latency, setLatency] = useState(0);
  const [testingTime, setTestingTime] = useState(0); // 0 - not tested yet
  // 1 - testing, 2 - tested
  const path = "weights/action/fc_net.all.ort";

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
    // let testTensor = new Ort.Tensor('float32',
    //   new Float32Array(joints * channels), [1, joints, channels]);
    let testTensor = new Ort.Tensor('float32',
      new Float32Array(
        [0.13606200000000002, 0.06848499999999999, 0.680415, 0.185593, 0.0, 0.683958, 0.11355799999999999, 0.010531000000000013, 0.69504, 0.25846, 
          0.06234100000000001, 0.685135, 0.09200600000000003, 0.047554999999999986, 0.667662, 0.313164, 0.332662, 0.66281, 0.11125600000000002, 0.35753900000000005, 0.643743, 0.36067400000000005, 0.704736, 0.691455, 0.11668600000000001, 0.745959, 0.695636, 0.08238200000000001, 0.826021, 
          0.642414, 0.0, 0.811585, 0.597152, 0.30617400000000006, 0.8493959999999999, 0.351919, 0.15309100000000003, 0.828836, 0.656354]), [1, joints, channels]);
    // console.log(testTensor);
    try {
      // console.log(testTensor);
      let times = new Array();
      setTestingTime(1);
      for (let i = 0; i < testTime; i++) {
        const t0 = performance.now();
        let objInput = {};
        objInput["input"] = testTensor;
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
