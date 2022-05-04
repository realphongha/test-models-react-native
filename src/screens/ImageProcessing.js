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
import * as jpeg from 'jpeg-js';
import * as RNFS from 'react-native-fs';
import {Buffer} from 'buffer';
import ImageResizer from 'react-native-image-resizer';
import { LogBox } from 'react-native';

LogBox.ignoreLogs([
  'Non-serializable values were found in the navigation state',
]);

export default function ImageProcessing({navigation, route}) {

  const [img, setImg] = useState(null);
  const std = [57.375, 57.12, 58.395];
  const mean = [103.53, 116.28, 123.675];

  useEffect(() => {
    console.log("Image path:", route.params.img.path);
  }, []);

  const testModel = async (path, nav, w, h, quality) => {
    let new_path;
    await ImageResizer.createResizedImage(path, w, h, 'JPEG', 
      quality, 0, undefined, false, {mode: 'stretch'})
      .then(response => {
        // response.uri is the URI of the new image that can now be displayed, uploaded...
        // response.path is the path of the new image
        // response.name is the name of the new image with the extension
        // response.size is the size of the new image
        new_path = response.uri;
        setImg(response.uri);
      })
      .catch(err => {
        console.log("Error resizing image:", err);
        // Oops, something went wrong. Check that the filename is correct and
        // inspect err to get more details.
      });
    // console.log(new_path);
    const imgBase64 = await RNFS.readFile(new_path, 'base64');
    const imgBuffer = Buffer.from(imgBase64, 'base64');
    const {width, height, data} = jpeg.decode(imgBuffer, {useTArray: true});
    // console.log(width, height, typeof data);
    const buffer = new Float32Array(width * height * 3);
    let offset = 0;
    for (let i = 0; i < buffer.length; i += 3) {
      buffer[i] = data[offset];
      buffer[i + 1] = data[offset + 1];
      buffer[i + 2] = data[offset + 2];
      offset += 4;
    }
    navigation.navigate(nav, {
      "modelInput": buffer,
    })
  }

  return (
    <View style={styles.container}>
      {img ?
      <Image
        style={styles.img}
        source={{
            uri: img,
        }}
      /> : 
        route.params.img.path ?
          <Image
            style={styles.img}
            source={{
                uri: "file://" + route.params.img.path,
            }}
          /> : 
          <Text style={styles.text}>Error loading image</Text>
      }
      <Button
        style={styles.btn}
        onPress={() => {testModel("file://" + route.params.img.path, 
          "NanoDet", 320, 320, 100)}}
        title="Test NanoDet"
      />
      <Button
        style={styles.btn}
        onPress={() => {testModel("file://" + route.params.img.path, 
          "UdpPose", 192, 256, 100)}}
        title="Test UdpPose"
      />
      <Button
        style={styles.btn}
        onPress={() => {testModel("file://" + route.params.img.path, 
          "FullPipeline", 320, 320, 100)}}
        title="Test Full Pipeline"
      />
      <Button
        style={styles.btn}
        onPress={() => {navigation.navigate("CamEngine")}}
        title="Back to Camera"
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
  btn: {
    paddingTop: 70,
  },
  text: {
    alignSelf: 'stretch',
    color: '#000',
  },
  img: {
    flex: 1,
    alignSelf: 'stretch',
  }
});
