import React from 'react';
import ModelAbs from './ModelSpeedTestAbs';

export default function UdpPose() {

  return (
    <ModelAbs
      w={192} h={256} 
      testTime={50}
      modelPath={"weights/pose/udp_shufflenetv2_plus_pixel_shuffle.ort"}
      inputName={"images"}
    />
  );
}
