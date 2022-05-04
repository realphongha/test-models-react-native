package com.testmodel.anticheatingmodels.models;

import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Log;

import com.testmodel.anticheatingmodels.utils.*;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ai.onnxruntime.*;

class SortByConf implements Comparator<float[]> {

    // descending sort:
    public int compare(float[] a, float[] b) {
        return Float.compare(b[4], a[4]);
    }
}

public class NanoDet {

    private final OrtSession session;
    private final OrtEnvironment env;

    public NanoDet(OrtSession session, OrtEnvironment env) {
        this.session = session;
        this.env = env;
    }

    private FloatBuffer preProcess(Bitmap bitmap) {
        return ImageUtils.preProcess(bitmap, Constants.OBJ_DET_INPUT_W, Constants.OBJ_DET_INPUT_H,
                false);
    }

    private void distance2bbox(float[] bbox, float[] dflPred, int x, int y, int stride){
        int ctX = x * stride;
        int ctY = y * stride;
        float[] disPred = {0f, 0f, 0f, 0f};
        int len = Constants.REG_MAX + 1;
        for (int i = 0; i < 4; i++){
            float dis = 0f;
            float[] disAfterSm = new float[len];
            float sum = 0f;
            int idx = i * len;
            float alpha = Utils.findLargestFloatArr(dflPred, idx, idx + len);
            for (int j = 0; j < len; j++) {
                disAfterSm[j] = (float) Math.exp((double) dflPred[idx + j] - alpha);
                sum += disAfterSm[j];
            }
            for (int j = 0; j < len; j++){
                dis += (disAfterSm[j] / sum * j);
            }
            dis *= stride;
            disPred[i] = dis;
        }
        bbox[0] = Math.max(ctX - disPred[0], 0);
        bbox[1] = Math.max(ctY - disPred[1], 0);
        bbox[2] = Math.min(ctX + disPred[2], Constants.OBJ_DET_INPUT_W);
        bbox[3] = Math.min(ctY + disPred[3], Constants.OBJ_DET_INPUT_H);
    }

    private float iouCalc(float[] box1, float[] box2){
        float box1Area = (box1[2]-box1[0])*(box1[3]-box1[1]);
        float box2Area = (box2[2]-box2[0])*(box2[3]-box2[1]);
        float[] leftUp = {Math.max(box1[0], box2[0]), Math.max(box1[1], box2[1])};
        float[] rightDown = {Math.min(box1[2], box2[2]), Math.min(box1[3], box2[3])};
        float[] intersect = {rightDown[0] - leftUp[0], rightDown[1] - leftUp[1]};
        intersect[0] = Math.max(intersect[0], 0f);
        intersect[1] = Math.max(intersect[1], 0f);
        float intersectArea = intersect[0] * intersect[1];
        // return 1.0*intersectArea/(box1Area+box2Area-intersectArea); // overlap
        return intersectArea / (Math.min(box1Area, box2Area)); // min size
    }

    private List<float[]> multiclassNms(List<float[]> bboxes){
        if (bboxes.size() == 0){
            return new ArrayList<float[]>();
        }
        bboxes.sort(new SortByConf());
        List<float[]> returnBboxes = new ArrayList<>();
        Map<Integer, List<float[]>> boxesMap = new HashMap<>();
        for (float[] box: bboxes){
            int cls = (int) Math.round(box[5]);
            if (boxesMap.containsKey(cls)){
                boxesMap.get(cls).add(box);
            } else {
                List<float[]> newList = new ArrayList<>();
                newList.add(box);
                boxesMap.put(cls, newList);
            }
        }
        boxesMap.forEach((cls, boxs) -> {
            if (boxs.size() == 1){
                returnBboxes.add(boxs.get(0));
            } else {
                while (boxs.size() != 0){
                    float[] bestBox = boxs.remove(0);
                    returnBboxes.add(bestBox);
                    for (int i = 0; i < boxs.size(); i++){
                        if (iouCalc(bestBox, boxs.get(i)) > Constants.IOU_THRESH){
                            boxs.remove(i);
                            i--;
                        }
                    }
                }
            }
        });
        return returnBboxes;
    }

    private List<float[]> postProcess(float[][] output) {
        List<float[]> results = new ArrayList<>();
        int numCands = output.length;
        int numFeats = output[0].length;
        int disFeatures = numFeats - Constants.OBJ_DET_NUM_CLS;
        int numCls = Constants.OBJ_DET_NUM_CLS;
        float[][] clsPreds = new float[numCands][numCls];
        float[][] disPreds = new float[numCands][disFeatures];
        for (int i = 0; i < numCands; i++){
            for (int j = 0; j < numFeats; j++){
                if (j < numCls){
                    clsPreds[i][j] = output[i][j];
                } else {
                    disPreds[i][j - numCls] = output[i][j];
                }
            }
        }
        List<int[]> centerPriors = new ArrayList<>();
        for (int stride: Constants.STRIDES){
            int featW = (int) Math.round(Math.ceil((double) Constants.OBJ_DET_INPUT_W/stride));
            int featH = (int) Math.round(Math.ceil((double) Constants.OBJ_DET_INPUT_H/stride));
            for (int y = 0; y < featH; y++){
                for (int x = 0; x < featW; x++){
                    centerPriors.add(new int[]{x, y, stride});
                }
            }
        }
        for (int i = 0; i < centerPriors.size(); i++) {
            int x = centerPriors.get(i)[0];
            int y = centerPriors.get(i)[1];
            int stride = centerPriors.get(i)[2];
            int maxCls = Utils.findLargestFloat(clsPreds[i]);
            float score = clsPreds[i][maxCls];
            if (score > Constants.SCORE_THRESH[maxCls]){
                float[] bbox = new float[6];
                distance2bbox(bbox, disPreds[i], x, y, stride);
                bbox[4] = score;
                bbox[5] = (float) maxCls;
                results.add(bbox);
            }
        }
        return multiclassNms(results);
    }

    public List<float[]> run(Bitmap bitmap, long t0) throws OrtException {
        long t1 = SystemClock.uptimeMillis();
        FloatBuffer input = preProcess(bitmap);
        long t2 = SystemClock.uptimeMillis();
        long[] shape = {1L, 3L, (long) Constants.OBJ_DET_INPUT_H, (long) Constants.OBJ_DET_INPUT_W};
        OnnxTensor tensor = OnnxTensor.createTensor(env, input, shape);
        OrtSession.Result result = session.run(Collections.singletonMap(
                Constants.OBJ_DET_INPUT_NAME, tensor));
        float[][] rawOutput = ((float[][][]) result.get(0).getValue())[0];
        long t3 = SystemClock.uptimeMillis();
        List<float[]> resultBoxes = postProcess(rawOutput);
        long t4 = SystemClock.uptimeMillis();
        Log.i("ReactNative", "Object detection speed:");
        Log.i("ReactNative", "Process image: " + String.valueOf(t1-t0) + " ms");
        Log.i("ReactNative", "Preprocess: " + String.valueOf(t2-t1) + " ms");
        Log.i("ReactNative", "Model: " + String.valueOf(t3-t2) + " ms");
        Log.i("ReactNative", "Postprocess: " + String.valueOf(t4-t3) + " ms");
        return resultBoxes;
    }
}
