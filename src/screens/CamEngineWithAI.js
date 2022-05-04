import React, {useState, useEffect, useRef} from 'react';
import {
  Text,
  View,
  TouchableWithoutFeedback,
  StyleSheet,
  Button,
  Image,
} from 'react-native';
import {Camera, useCameraDevices, useFrameProcessor} from 'react-native-vision-camera';
import KeepAwake from 'react-native-keep-awake';
import * as constants from '../utils/Constant';
import {antiCheatingModels} from '../types/frameProcessorUtils'


export default function CamEngineWithAI({navigation}) {
  const [perm, setPerm] = useState(null);
  const [camPosition, setCamPosition] = useState("back");
  const devices = useCameraDevices();
  const cameraRef = useRef(null);
  const actionStr = {
    0: "Hand reach out",
    1: "Look down",
    2: "Look outside", 
    3: "Sitting"
  }

  useEffect(() => {
    initCam();
  }, []);

  const initCam = async () => {
    const newCameraPermission = await Camera.requestCameraPermission();
    setPerm(newCameraPermission === "authorized");
    console.log("Device:", devices[camPosition]);
    if (devices[camPosition]) {
      console.log("Cam ID:", devices[camPosition].id);
    }
    console.log("Permission:", perm);
  };

  const onFlip = () => {
    setCamPosition(camPosition==="front"?"back":"front");
  };

  const onSnap = async () => {
    let options = {
      flash: 'auto',
      qualityPrioritization: 'speed',
      skipMetadata: false,
    }
    let img = await cameraRef.current.takePhoto(options);
    // img.path = "/data/user/0/com.testmodel/cache/blah.jpg";
    navigation.navigate("ImageProcessing", {
      "img": img,
    })
  };

  const onShowObj = async () => {
    navigation.navigate("DisplayImage", {
      imgUri: "file://" + "/data/user/0/com.testmodel/cache/obj_det.jpg"
    })
  };

  const onShowPose = async () => {
    navigation.navigate("DisplayImage", {
      imgUri: "file://" + "/data/user/0/com.testmodel/cache/pose.jpg"
    })
  };

  const frameProcess = useFrameProcessor(async (frame) => {
    'worklet';
    let result = antiCheatingModels(frame);
    if (result.length == 6){
    console.log(actionStr[result[4]], result[5][result[4]]);
    } else {
      console.log("No person!");
    }
    // setPhoto({path: "file://" + result});
    // navigation.navigate("DisplayImage", {
    //   "imgUri": result,
    // })
  }, [])

  return (
    <View style={styles.container}>
      <KeepAwake />
      {perm && devices[camPosition] && (
        <Camera
          ref={cameraRef}
          style={styles.camera}
          device={devices[camPosition]}
          isActive={true}
          photo={true}
          frameProcessor={frameProcess}
          frameProcessorFps={2}
          // preset="low"
        />
      )}
      {perm && devices[camPosition] && (
        <Button style={styles.btn} onPress={() => onFlip()} title="Flip!" />
      )}
      {perm && devices[camPosition] && 
        <Button style={styles.btn} onPress={() => onSnap()} title="Snap!" />
      }
      {perm && devices[camPosition] && 
        <Button style={styles.btn} onPress={() => onShowObj()} title="Show objects!" />
      } 
      {perm && devices[camPosition] && 
        <Button style={styles.btn} onPress={() => onShowPose()} title="Show pose!" />
      } 
      <Button
        style={styles.btn}
        onPress={() => initCam()}
        title="Re-init Camera"
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignSelf: 'stretch',
    backgroundColor: constants.gray,
    flexDirection: 'column',
    alignItems: 'flex-end',
    justifyContent: 'flex-end'
  },
  camera: StyleSheet.absoluteFill,
  btn: {
    paddingTop: 70,
  },
  text: {
    color: '#000',
  },
  img: {
    flex: 1,
    alignSelf: 'stretch',
  }
});
