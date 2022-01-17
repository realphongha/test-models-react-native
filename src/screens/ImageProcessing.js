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

export default function ImageProcessing({navigation, route}) {

  useEffect(() => {
    console.log("Image path:", route.params.img.path);
  }, []);

  return (
    <View style={styles.container}>
      {route.params.img.path ?
      <Image
        style={styles.img}
        source={{
            uri: "file://" + route.params.img.path,
        }}
      /> : 
      <Text style={styles.text}>Error loading image</Text>}
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
