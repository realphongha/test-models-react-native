import React from 'react';
import {StyleSheet, Text, View, Button} from 'react-native';
import { createStackNavigator } from '@react-navigation/stack';
import { NavigationContainer } from '@react-navigation/native';
import CamEngine from './src/screens/CamEngine';
import CamEngineWithAI from './src/screens/CamEngineWithAI'
import ImageProcessing from './src/screens/ImageProcessing';
import SpeedTestScreen from './src/screens/speedtest/SpeedTestScreen';
import NanoDet from './src/screens/NanoDet';
import UdpPose from './src/screens/UdpPose';
import FullPipeline from './src/screens/FullPipeline';
import DisplayImage from './src/screens/DisplayImage';

const Welcome = ({navigation}) => {
  return (
    <View>
      <Button 
        style={styles.btn}
        title="Cam engine"
        onPress={() => navigation.navigate("CamEngine")}
        />
      <Button 
        style={styles.btn}
        title="Cam engine with AI"
        onPress={() => navigation.navigate("CamEngineWithAI")}
        />
      <Button 
        style={styles.btn}
        title="Test model speed"
        onPress={() => navigation.navigate("SpeedTestScreen")}  
        />
    </View>
  )
}

const Stack = createStackNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator initialRouteName="Welcome">
        <Stack.Screen name="Welcome" component={Welcome} />
        <Stack.Screen name="CamEngine" component={CamEngine} />
        <Stack.Screen name="CamEngineWithAI" component={CamEngineWithAI} />
        <Stack.Screen name="ImageProcessing" component={ImageProcessing} />
        <Stack.Screen name="SpeedTestScreen" component={SpeedTestScreen} />
        <Stack.Screen name="DisplayImage" component={DisplayImage} />
        <Stack.Screen name="NanoDet" component={NanoDet} />
        <Stack.Screen name="UdpPose" component={UdpPose} />
        <Stack.Screen name="FullPipeline" component={FullPipeline} />
      </Stack.Navigator>
    </NavigationContainer>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  btn: {
    paddingTop: 70,
  }
});
