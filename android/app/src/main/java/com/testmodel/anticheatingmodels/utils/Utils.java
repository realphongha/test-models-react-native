package com.testmodel.anticheatingmodels.utils;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.YuvImage;

import androidx.camera.core.ImageProxy;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;

public class Utils {
    public static byte[] inputStreamToByteArr(InputStream is) throws IOException {
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        int nRead;
        byte[] data = new byte[20971520];
        while ((nRead = is.read(data, 0, data.length)) != -1) {
            buffer.write(data, 0, nRead);
        }
        return buffer.toByteArray();
    }

    public static int findLargestInt(int[] arr)
    {
        int largest = arr[0];
        int largestIndex = 0;

        for(int i = 1; i < arr.length; i++)
        {
            if(arr[i] > largest) {
                largest = arr[i];
                largestIndex = i;
            }
        }
        return largestIndex;
    }

    public static int findLargestFloat(float[] arr) {
        float largest = arr[0];
        int largestIndex = 0;

        for(int i = 1; i < arr.length; i++)
        {
            if(arr[i] > largest) {
                largest = arr[i];
                largestIndex = i;
            }
        }
        return largestIndex;
    }

    public static float findLargestFloatArr(float[] arr, int start, int end) {
        float largest = arr[start];
        for (int i = start + 1; i < end; i++){
            if (arr[i] > largest) {
                largest = arr[i];
            }
        }
        return largest;
    }

    public static float[] softmax(float[] arr){
        float[] returnArr = new float[arr.length];
        float sum = 0f;
        for (int i = 0; i < arr.length; i++){
            returnArr[i] = (float) Math.exp((double) arr[i]);
            sum += returnArr[i];
        }
        for (int i = 0; i < arr.length; i++) {
            returnArr[i] /= sum;
        }
        return returnArr;
    }

}
