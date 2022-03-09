import React from 'react';
import ModelAbs from './ModelSpeedTestAbs';

export default function NanoDetMO() {

  return (
    <ModelAbs
      w={320} h={320} 
      testTime={50}
      modelPath={"weights/objdet/nanodet_plus_m_320.all.ort"} /* nanodet_plus_m_320.all.ort, picodet_s_320_coco_exam_b64.all.ort, yolov5n_320_coco_exam_from_scratch.all.ort */
      inputName={"image"} /* data, image, images */
    />
  );
}