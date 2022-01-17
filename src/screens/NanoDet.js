import React from 'react';
import ModelAbs from './ModelSpeedTestAbs';

export default function UdpPose() {

  return (
    <ModelAbs
      w={320} h={320} 
      testTime={50}
      modelPath={"weights/objdet/nanodet_plus_m_320.ort"}
      inputName={"data"}
    />
  );
}