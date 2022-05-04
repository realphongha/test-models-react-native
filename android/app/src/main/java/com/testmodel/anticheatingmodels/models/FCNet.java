package com.testmodel.anticheatingmodels.models;

import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Log;

import com.testmodel.anticheatingmodels.utils.Constants;
import com.testmodel.anticheatingmodels.utils.PointUtils;
import com.testmodel.anticheatingmodels.utils.Utils;

import java.nio.FloatBuffer;
import java.util.Collections;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

public class FCNet {

    private final OrtSession session;
    private final OrtEnvironment env;

    public FCNet(OrtSession session, OrtEnvironment env) {
        this.session = session;
        this.env = env;
    }

    private FloatBuffer preProcess(float[][] pose) {
        FloatBuffer inputBuffer = FloatBuffer.allocate(Constants.ACTION_INPUT_JOINTS *
                Constants.ACTION_INPUT_CHANNELS);
        double rad = PointUtils.angle((pose[6][0]+pose[5][0]-pose[8][0]-pose[7][0])/2,
                (pose[6][1]+pose[5][1]-pose[8][1]-pose[7][1])/2);
        float[][] poseRotated = PointUtils.rotateKps(pose, rad);
        float min0 = 9999999;
        float max0 = -9999999;
        float min1 = 9999999;
        float max1 = -9999999;
        for (float[] point: poseRotated){
            if (point[0] < min0) min0 = point[0];
            if (point[0] > max0) max0 = point[0];
            if (point[1] < min1) min1 = point[1];
            if (point[1] > max1) max1 = point[1];
        }
        inputBuffer.rewind();
        int idx = 0;
        for (float[] point: poseRotated){
            inputBuffer.put(idx++, (point[0] - min0) / (max0 - min0));
            inputBuffer.put(idx++, (point[1] - min1) / (max1 - min1));
            if (Constants.ACTION_INPUT_CHANNELS == 3) {
                inputBuffer.put(idx++, point[2]);
            }
        }
        inputBuffer.rewind();
        return inputBuffer;
    }

    private float[] postProcess(float[] output) {
        return Utils.softmax(output);
    }

    public float[] run(float[][] pose, long t0) throws OrtException {
        long t1 = SystemClock.uptimeMillis();
        FloatBuffer input = preProcess(pose);
        long t2 = SystemClock.uptimeMillis();
        long[] shape = {1L,
                (long) Constants.ACTION_INPUT_JOINTS, (long) Constants.ACTION_INPUT_CHANNELS};
        OnnxTensor tensor = OnnxTensor.createTensor(env, input, shape);
        OrtSession.Result result = session.run(Collections.singletonMap(
                Constants.ACTION_INPUT_NAME, tensor));
        float[] rawOutput = ((float[][]) result.get(0).getValue())[0];
        long t3 = SystemClock.uptimeMillis();
        float[] prob = postProcess(rawOutput);
        long t4 = SystemClock.uptimeMillis();
        Log.i("ReactNative", "Action speed:");
        Log.i("ReactNative", "Process image: " + String.valueOf(t1-t0) + " ms");
        Log.i("ReactNative", "Preprocess: " + String.valueOf(t2-t1) + " ms");
        Log.i("ReactNative", "Model: " + String.valueOf(t3-t2) + " ms");
        Log.i("ReactNative", "Postprocess: " + String.valueOf(t4-t3) + " ms");
        return prob;
    }
}
