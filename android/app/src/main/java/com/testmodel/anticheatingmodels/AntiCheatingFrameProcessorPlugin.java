package com.testmodel.anticheatingmodels;

import androidx.annotation.NonNull;
import androidx.camera.core.ImageProxy;

import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.WritableNativeArray;
import com.mrousavy.camera.frameprocessor.FrameProcessorPlugin;
import com.testmodel.anticheatingmodels.models.*;
import com.testmodel.anticheatingmodels.utils.*;

import android.annotation.SuppressLint;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Paint;
import android.graphics.Rect;
import android.media.AudioManager;
import android.media.Image;
import android.media.ToneGenerator;
import android.os.SystemClock;
import android.util.Log;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.List;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;
import ai.onnxruntime.OrtSession.RunOptions;
import ai.onnxruntime.OrtSession.SessionOptions;

public class AntiCheatingFrameProcessorPlugin extends FrameProcessorPlugin {

    private final OrtEnvironment env = OrtEnvironment.getEnvironment();
    private OrtSession sessionObjDet = null;
    private OrtSession sessionPose = null;
    private OrtSession sessionAction = null;
    private NanoDet nanodetEngine = null;
    private UdpPose udpEngine = null;
    private FCNet fcnetEngine = null;
    private ReactApplicationContext context = null;
    private final int PERSON_PADDING = 5;

    AntiCheatingFrameProcessorPlugin(ReactApplicationContext context) {
        super("antiCheatingModels");
        Resources resources = context.getResources();
        this.context = context;
        Log.i("ReactNative", "init");
        try {
            InputStream objDetWeights = resources.getAssets().open(Constants.OBJ_DET_WEIGHTS);
            InputStream poseWeights = resources.getAssets().open(Constants.POSE_WEIGHTS);
            InputStream actionWeights = resources.getAssets().open(Constants.ACTION_WEIGHTS);
            sessionObjDet = env.createSession(Utils.inputStreamToByteArr(objDetWeights));
            sessionPose = env.createSession(Utils.inputStreamToByteArr(poseWeights));
            sessionAction = env.createSession(Utils.inputStreamToByteArr(actionWeights));
            Log.i("ReactNative", sessionObjDet.toString());
            Log.i("ReactNative", sessionPose.toString());
            Log.i("ReactNative", sessionAction.toString());
            nanodetEngine = new NanoDet(sessionObjDet, env);
            udpEngine = new UdpPose(sessionPose, env);
            fcnetEngine = new FCNet(sessionAction, env);
        } catch (OrtException e) {
            Log.e("ReactNative", "Can't create InferenceSession");
            Log.e("ReactNative", e.toString());
        } catch (IOException e) {
            Log.e("ReactNative", e.toString());
        }
    }

    @SuppressLint("UnsafeOptInUsageError")
    @Override
    public Object callback(@NonNull ImageProxy frame, @NonNull Object[] params) {
        WritableNativeArray results = new WritableNativeArray();
        try {
            long t0 = SystemClock.uptimeMillis();
            Bitmap imgBitmap = ImageUtils.toBitmap(frame);
            Bitmap rawBitmap = Bitmap.createScaledBitmap(imgBitmap,
                    Constants.OBJ_DET_INPUT_W, Constants.OBJ_DET_INPUT_H, false);
            Bitmap bitmap = null;
            if (rawBitmap != null) {
                bitmap = ImageUtils.rotate(rawBitmap, (float) frame.getImageInfo().getRotationDegrees());
            } else {
                throw new Exception("Cannot resize bitmap!");
            }
            if (bitmap == null) {
                throw new Exception("Cannot rotate bitmap!");
            }
            List<float[]> boxes = nanodetEngine.run(bitmap, t0);
            Canvas canvas = new Canvas(bitmap);
            Log.i("ReactNative", "Detected " + String.valueOf(boxes.size()) + " boxes:");
            WritableNativeArray bboxes = new WritableNativeArray();
            boolean foundPerson = false;
            float[] bestPersonBox = null;
            for (float[] box: boxes){
                Log.i("ReactNative", Arrays.toString(box));
                WritableNativeArray returnBox = new WritableNativeArray();
                int x0 = (int) Math.round(box[0]);
                int y0 = (int) Math.round(box[1]);
                int x1 = (int) Math.round(box[2]);
                int y1 = (int) Math.round(box[3]);
                int cls = (int) Math.round(box[5]);
                returnBox.pushInt(x0);
                returnBox.pushInt(y0);
                returnBox.pushInt(x1);
                returnBox.pushInt(y1);
                returnBox.pushDouble((double) box[4]);
                returnBox.pushInt(cls);
                bboxes.pushArray(returnBox);

                if (cls == Constants.PERSON_CLS){
                    if (foundPerson) {
                        if (box[4] > bestPersonBox[4]){
                            bestPersonBox = new float[]{x0, y0, x1, y1, box[4]};
                        }
                    } else {
                        bestPersonBox = new float[]{x0, y0, x1, y1, box[4]};
                        foundPerson = true;
                    }
                }
            }
            Bitmap cropBitmap = null;
            Bitmap resizedBitmap = null;
            Bitmap rotatedBitmap = null;
            if (foundPerson) {
//                float newX0 = Math.max(bestPersonBox[0] / Constants.OBJ_DET_INPUT_W *
//                        imgBitmap.getWidth() - PERSON_PADDING, 0f);
//                float newY0 = Math.max(bestPersonBox[1] / Constants.OBJ_DET_INPUT_H *
//                        imgBitmap.getHeight() - PERSON_PADDING, 0f);
//                float newX1 = Math.min(bestPersonBox[2] / Constants.OBJ_DET_INPUT_W *
//                        imgBitmap.getWidth() + PERSON_PADDING, imgBitmap.getWidth());
//                float newY1 = Math.min(bestPersonBox[3] / Constants.OBJ_DET_INPUT_H *
//                        imgBitmap.getHeight() + PERSON_PADDING, imgBitmap.getHeight());
//                cropBitmap = Bitmap.createBitmap(imgBitmap,
//                        (int) newX0, (int) newY0,
//                        (int) (newX1 - newX0), (int) (newY1 - newY0));
//                resizedBitmap = Bitmap.createScaledBitmap(cropBitmap,
//                        Math.max(Constants.POSE_INPUT_H, Constants.POSE_INPUT_W),
//                        Math.max(Constants.POSE_INPUT_H, Constants.POSE_INPUT_W),
//                        false);
//                rotatedBitmap = ImageUtils.rotate(resizedBitmap,
//                        frame.getImageInfo().getRotationDegrees());
                float newX0 = Math.max(bestPersonBox[0] - PERSON_PADDING, 0f);
                float newY0 = Math.max(bestPersonBox[1] - PERSON_PADDING, 0f);
                float newX1 = Math.min(bestPersonBox[2] + PERSON_PADDING, Constants.OBJ_DET_INPUT_W);
                float newY1 = Math.min(bestPersonBox[3] + PERSON_PADDING, Constants.OBJ_DET_INPUT_H);
                cropBitmap = Bitmap.createBitmap(bitmap,
                        (int) newX0, (int) newY0,
                        (int) (newX1 - newX0), (int) (newY1 - newY0));
//                resizedBitmap = Bitmap.createScaledBitmap(cropBitmap,
//                        Math.max(Constants.POSE_INPUT_H, Constants.POSE_INPUT_W),
//                        Math.max(Constants.POSE_INPUT_H, Constants.POSE_INPUT_W),
//                        false);
//                rotatedBitmap = ImageUtils.rotate(resizedBitmap,
//                        frame.getImageInfo().getRotationDegrees());
            }

            // draw obj:
            for (float[] box: boxes){
                int x0 = (int) Math.round(box[0]);
                int y0 = (int) Math.round(box[1]);
                int x1 = (int) Math.round(box[2]);
                int y1 = (int) Math.round(box[3]);
                int cls = (int) Math.round(box[5]);

                // draw obj box:
                Paint paint = new Paint();
                paint.setStyle(Paint.Style.STROKE);
                paint.setColor(Color.RED);
                paint.setAntiAlias(true);
                Rect rect = new Rect(x0, y0, x1, y1);
                canvas.drawRect(rect, paint);
                paint.setTextSize(20);
                paint.setStyle(Paint.Style.FILL);
                canvas.drawText(Constants.OBJ_DET_CLASSES[cls],
                        x0, y0, paint);
            }
            results.pushArray(bboxes);
            String filename = context.getCacheDir().getAbsolutePath() + "/obj_det.jpg";
            File file = new File(filename);
            if (file.exists()){
                file.delete();
            }
            FileOutputStream fileStream = new FileOutputStream(filename);
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, fileStream);
            fileStream.close();
            results.pushString(filename);

            if (!foundPerson){
                return results;
            }

            // pose:
            t0 = SystemClock.uptimeMillis();
            Bitmap personBitmap = Bitmap.createScaledBitmap(cropBitmap,
                    Constants.POSE_INPUT_W, Constants.POSE_INPUT_H,false);
            Canvas canvas1 = new Canvas(personBitmap);
            float[][] pose = udpEngine.run(personBitmap, t0);
            WritableNativeArray returnPose = new WritableNativeArray();
            for (float[] point: pose){
                returnPose.pushDouble((double) point[0]);
                returnPose.pushDouble((double) point[1]);
                returnPose.pushDouble((double) point[2]);

                // draw pose:
                Paint paint1 = new Paint();
                paint1.setStyle(Paint.Style.FILL);
                paint1.setColor(Color.RED);
                paint1.setAntiAlias(true);
                canvas1.drawCircle(point[0], point[1], 5, paint1);
            }
            results.pushArray(returnPose);
//            Log.i("ReactNative", "Pose:");
//            for (float[] point: pose){
//                Log.i("ReactNative", Arrays.toString(point));
//            }
            filename = context.getCacheDir().getAbsolutePath() + "/pose.jpg";
            File file2 = new File(filename);
            if (file2.exists()){
                file2.delete();
            }
            fileStream = new FileOutputStream(filename);
            personBitmap.compress(Bitmap.CompressFormat.JPEG, 100, fileStream);
            fileStream.close();
            results.pushString(filename);

            // action:
            t0 = SystemClock.uptimeMillis();
            float[] prob = fcnetEngine.run(pose, t0);
            int maxCls = Utils.findLargestFloat(prob);
            WritableNativeArray returnProb = new WritableNativeArray();
            for (float p: prob){
                returnProb.pushDouble((double) p);
            }
            results.pushInt(maxCls);
            results.pushArray(returnProb);
            Log.i("ReactNative", "Action result:");
            Log.i("ReactNative", Arrays.toString(prob));
            Log.i("ReactNative", Constants.ACTION_CLASSES[maxCls]);
            if (maxCls != Constants.SITTING_CLS && prob[maxCls] > 0.8) {
                ToneGenerator toneGen = new ToneGenerator(AudioManager.STREAM_MUSIC, 100);
                toneGen.startTone(ToneGenerator.TONE_CDMA_PIP, 100);
            }

        } catch (Exception e) {
            Log.e("ReactNative", e.toString());
        }
        return results;
    }
}
