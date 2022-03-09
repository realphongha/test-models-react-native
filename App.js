import React from 'react';
import {StyleSheet, Text, View, Button} from 'react-native';
import { createStackNavigator } from '@react-navigation/stack';
import { NavigationContainer } from '@react-navigation/native';
import CamEngine from './src/screens/CamEngine';
import ImageProcessing from './src/screens/ImageProcessing';
import NanoDet from './src/screens/NanoDet';
import UdpPose from './src/screens/UdpPose';
import SpeedTestScreen from './src/screens/speedtest/SpeedTestScreen';

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
        <Stack.Screen name="ImageProcessing" component={ImageProcessing} />
        <Stack.Screen name="SpeedTestScreen" component={SpeedTestScreen} />
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
