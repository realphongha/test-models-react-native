import React from 'react';
import {StyleSheet, Text, View} from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import NanoDet from '../NanoDet';
import UdpPose from '../UdpPose';
import { createStackNavigator } from '@react-navigation/stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';

const Stack = createStackNavigator();
const Tab = createBottomTabNavigator();

export default function SpeedTestScreen() {
  return (
    <NavigationContainer>
      <Tab.Navigator>
        <Tab.Screen name="NanoDet" component={NanoDet} />
        <Tab.Screen name="UdpPose" component={UdpPose}/>
      </Tab.Navigator>
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
});
