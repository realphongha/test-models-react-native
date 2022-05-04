import React, { useState, useEffect, useRef } from 'react';
import {
  Text,
  View,
  StyleSheet,
  Image,
} from 'react-native';
import * as constants from '../utils/Constant';

export default function DisplayImage({navigation, route}) {
  const [imgBase64, setImgBase64] = useState(null);
  const [imgUri, setImgUri] = useState(null);
  const [text, setText] = useState(null);

  useEffect(() => {
    if (route.params.imgBase64){
    setImgBase64(route.params.imgBase64);
    }
    if (route.params.text){
      setText(route.params.text);
    }
    if (route.params.imgUri){
      setImgUri(route.params.imgUri);
    }
  }, []);

  return (
    <View style={styles.container}>
      {imgBase64 && 
      <Image
        style={styles.img}
        source={{
            uri: `data:image/jpeg;base64,${imgBase64}`,
        }}
      />}
      {imgUri && 
      <Image
        style={styles.img}
        source={{
            uri: imgUri,
        }}
      />}
      {text &&
      <Text styles={styles.text}>
        {text}
      </Text>
      }
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
  text: {
    alignSelf: 'center',
    color: '#fff'
  },
  img: {
    flex: 1,
    alignSelf: 'stretch',
    maxHeight: 320,
    maxWidth: 320,
  }
});