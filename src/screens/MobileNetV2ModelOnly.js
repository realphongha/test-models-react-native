import React from 'react';
import ModelAbs from './ModelSpeedTestAbs';

export default function MobileNetV2MO() {

  return (
    <ModelAbs
      w={224} h={224} 
      testTime={50}
      modelPath={"weights/clf/mobilenet_v2_uint8.ort"} /* mobilenet_v2_float.ort */
      inputName={"input"}
    />
  );
}