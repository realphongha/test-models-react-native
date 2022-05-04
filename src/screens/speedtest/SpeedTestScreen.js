import React from 'react';
import {StyleSheet, Text, View} from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import NanoDetMO from '../NanoDetModelOnly';
import UdpPoseMO from '../UdpPoseModelOnly';
import FCNetMO from '../FCNetModelOnly';
import MobileNetV2MO from '../MobileNetV2ModelOnly';
import { createStackNavigator } from '@react-navigation/stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';

const Stack = createStackNavigator();
const Tab = createBottomTabNavigator();

export default function SpeedTestScreen() {
  return (
    <Tab.Navigator>
      <Tab.Screen name="NanoDetMO" component={NanoDetMO} />
      <Tab.Screen name="UdpPoseMO" component={UdpPoseMO}/>
      <Tab.Screen name="FCNetMO" component={FCNetMO}/>
      <Tab.Screen name="MobileNetV2MO" component={MobileNetV2MO}/>
    </Tab.Navigator>
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
