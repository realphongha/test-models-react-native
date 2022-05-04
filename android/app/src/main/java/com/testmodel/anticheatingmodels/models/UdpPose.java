package com.testmodel.anticheatingmodels.models;

import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Log;

import com.testmodel.anticheatingmodels.utils.Constants;
import com.testmodel.anticheatingmodels.utils.ImageUtils;
import com.testmodel.anticheatingmodels.utils.Utils;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

public class UdpPose {

    private final OrtSession session;
    private final OrtEnvironment env;

    public UdpPose(OrtSession session, OrtEnvironment env) {
        this.session = session;
        this.env = env;
    }

    private FloatBuffer preProcess(Bitmap bitmap) {
        return ImageUtils.preProcess(bitmap, Constants.POSE_INPUT_W, Constants.POSE_INPUT_H,
                true);
    }

    private float[][] postProcess(float[][][] output) {
        int hmH = output[0].length;
        int hmW = output[0][0].length;
        int hmSize = hmH * hmW;
        float[][] results = new float[Constants.ACTION_INPUT_JOINTS][3];
        float[][] newHeatmaps = new float[Constants.ACTION_INPUT_JOINTS][hmSize];
        for (int i = 0; i < Constants.ACTION_INPUT_JOINTS; i++){
            int idx = 0;
            for (int j = 0; j < hmH; j++) {
                for (int k = 0; k < hmW; k++){
                    newHeatmaps[i][idx++] = output[i][j][k];
                }
            }
        }
        for (int i = 0; i < Constants.ACTION_INPUT_JOINTS; i++){
            int maxIndex = Utils.findLargestFloat(newHeatmaps[i]);
            float score = newHeatmaps[i][maxIndex];
            results[i][2] = Math.max(score, 0f);
            if (score < 0f){
                results[i][0] = 0f;
                results[i][1] = 0f;
            } else {
                int x = maxIndex % hmW;
                int y = maxIndex / hmW;
                x = (int) Math.round((float) x / hmW * Constants.POSE_INPUT_W);
                y = (int) Math.round((float) y / hmH * Constants.POSE_INPUT_H);
                results[i][0] = (float) x;
                results[i][1] = (float) y;
            }
        }
        return results;
    }

    public float[][] run(Bitmap bitmap, long t0) throws OrtException {
        long t1 = SystemClock.uptimeMillis();
        FloatBuffer input = preProcess(bitmap);
        long t2 = SystemClock.uptimeMillis();
        long[] shape = {1L, 3L, (long) Constants.POSE_INPUT_H, (long) Constants.POSE_INPUT_W};
        OnnxTensor tensor = OnnxTensor.createTensor(env, input, shape);
        OrtSession.Result result = session.run(Collections.singletonMap(
                Constants.POSE_INPUT_NAME, tensor));
        float[][][] rawOutput = ((float[][][][]) result.get(0).getValue())[0];
        long t3 = SystemClock.uptimeMillis();
        float[][] pose = postProcess(rawOutput);
        long t4 = SystemClock.uptimeMillis();
        Log.i("ReactNative", "Pose speed:");
        Log.i("ReactNative", "Process image: " + String.valueOf(t1-t0) + " ms");
        Log.i("ReactNative", "Preprocess: " + String.valueOf(t2-t1) + " ms");
        Log.i("ReactNative", "Model: " + String.valueOf(t3-t2) + " ms");
        Log.i("ReactNative", "Postprocess: " + String.valueOf(t4-t3) + " ms");
        return pose;
    }
}
