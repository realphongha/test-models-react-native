package com.testmodel.anticheatingmodels.utils;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.util.Log;

import androidx.camera.core.ImageProxy;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;

public class PointUtils {

    public static FloatBuffer preProcess(Bitmap bitmap, int w, int h, boolean convertToBgr) {
        FloatBuffer imgData = FloatBuffer.allocate(3 * h * w);
        imgData.rewind();
        int stride = h * w;
        int[] bmpData = new int[stride];
        bitmap.getPixels(bmpData, 0, bitmap.getWidth(), 0, 0,
                bitmap.getWidth(), bitmap.getHeight());
        for (int i = 0; i < h; i++){
            for (int j = 0; j < w; j++){
                int idx = w * i + j;
                int pixelValue = bmpData[idx];
                if (convertToBgr) {
                    imgData.put(idx + stride * 2, (((pixelValue >> 16 & 0xFF) / 255f - 0.485f) / 0.229f));
                    imgData.put(idx + stride, (((pixelValue >> 8 & 0xFF) / 255f - 0.456f) / 0.224f));
                    imgData.put(idx, (((pixelValue & 0xFF) / 255f - 0.406f) / 0.225f));
                } else {
                    imgData.put(idx, (((pixelValue >> 16 & 0xFF) / 255f - 0.485f) / 0.229f));
                    imgData.put(idx + stride, (((pixelValue >> 8 & 0xFF) / 255f - 0.456f) / 0.224f));
                    imgData.put(idx + stride * 2, (((pixelValue & 0xFF) / 255f - 0.406f) / 0.225f));
                }
            }
        }
        imgData.rewind();
        return imgData;
    }

    public static double angle(double vx, double vy) {
        double cos = -vy / Math.sqrt(vx*vx + vy*vy);
        return Math.acos(cos);
    }

    public static float[][] rotateKps(float[][] kps, double rad){
        float meanX = 0.0f;
        float meanY = 0.0f;
        for (int i = 0; i < kps.length; i++){
            for (int j = 0; j < 2; j++){
                if (j == 0){
                    meanX += kps[i][j];
                } else {
                    meanY += kps[i][j];
                }
            }
        }
        meanX /= kps.length;
        meanY /= kps.length;

        float[][] kpsNew = new float[kps.length][kps[0].length];
        for (int i = 0; i < kps.length; i++){
            for (int j = 0; j < 2; j++){
                kpsNew[i][j] = kps[i][j] - ((j == 0)?meanX:meanY);
            }
        }
        float[][] kpsRet = new float[kps.length][kps[0].length];
        for (int i = 0; i < kps.length; i++){
            for (int j = 0; j < 2; j++){
                if (j == 0) {
                    kpsRet[i][j] = (float) (Math.cos(rad) * kpsNew[i][0] - Math.sin(rad) * kpsNew[i][1] + meanX);
                } else {
                    kpsRet[i][j] = (float) (Math.sin(rad) * kpsNew[i][0] + Math.cos(rad) * kpsNew[i][1] + meanY);
                }
            }
        }
        return kpsRet;
    }
}
