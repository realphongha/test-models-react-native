import React from 'react';
import {StyleSheet, Text, View} from 'react-native';
import { createStackNavigator } from '@react-navigation/stack';
import { NavigationContainer } from '@react-navigation/native';
import CamEngine from './src/screens/CamEngine';
import ImageProcessing from './src/screens/ImageProcessing';
import NanoDet from './src/screens/NanoDet';
import UdpPose from './src/screens/UdpPose';
import SpeedTestScreen from './src/screens/speedtest/SpeedTestScreen';

const Stack = createStackNavigator();

export default function App() {
  return (
    <SpeedTestScreen />
    //<NavigationContainer>
      //<Stack.Navigator>
        //<Stack.Screen name="CamEngine" component={CamEngine} />
        //<Stack.Screen name="ImageProcessing" component={ImageProcessing}/>
      //</Stack.Navigator>
    //</NavigationContainer>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
});
