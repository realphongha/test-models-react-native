import React, {useState, useEffect, useRef} from 'react';
import {
  Text,
  View,
  TouchableWithoutFeedback,
  StyleSheet,
  Button,
  Image,
} from 'react-native';
import {Camera, useCameraDevices} from 'react-native-vision-camera';
import * as constants from '../utils/Constant';

export default function CamEngine({navigation}) {
  const [perm, setPerm] = useState(null);
  const [camPosition, setCamPosition] = useState("back");
  const devices = useCameraDevices();
  const cameraRef = useRef(null);
  const [photo, setPhoto] = useState(null);
  const [ratio, setRatio] = useState(null);

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

  const initRatio = async () => {};

  const onFlip = () => {
    setCamPosition(camPosition==="front"?"back":"front");
  };

  const onSnap = async () => {
    let options = {
      flash: 'auto',
    }
    let img = await cameraRef.current.takePhoto(options);
    // setPhoto(img.path);
    // console.log(img);
    // console.log(img);
    navigation.navigate("ImageProcessing", {
      "img": img,
    })
  };

  if (photo){
    return (
      <View style={styles.container}>
          <Image
            style={styles.img}
            source={{
              uri: photo.path,
            }}
          />
        
        <Button
          style={styles.btn}
          onPress={setPhoto(null)}
          title="Back to Camera"
        />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {perm && devices[camPosition] && (
        <Camera
          ref={cameraRef}
          style={styles.camera}
          device={devices[camPosition]}
          isActive={true}
          photo={true}
        />
      )}
      {perm && devices[camPosition] && (
        <Button style={styles.btn} onPress={() => onFlip()} title="Flip!" />
      )}
      {perm && devices[camPosition] && 
        <Button style={styles.btn} onPress={() => onSnap()} title="Snap!" />
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
