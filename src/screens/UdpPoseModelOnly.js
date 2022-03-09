import React from 'react';
import ModelAbs from './ModelSpeedTestAbs';

export default function UdpPoseMO() {

  return (
    <ModelAbs
      w={192} h={256} 
      testTime={50}
      modelPath={"weights/pose/pose_shufflenetv2_plus_pixel_shuffle.all.ort"} /* pose_shufflenetv2_plus_pixel_shuffle.all.ort, pose_shufflenetv2_10x_pixel_shuffle.all.ort, pose_mobilenetv3_small_pixel_shuffle.all.ort */
      inputName={"images"}
    />
  );
}
