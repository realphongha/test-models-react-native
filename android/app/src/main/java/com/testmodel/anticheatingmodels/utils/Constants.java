package com.testmodel.anticheatingmodels.utils;

public class Constants {
    // model inputs:
    public static final int OBJ_DET_INPUT_W = 320;
    public static final int OBJ_DET_INPUT_H = 320;
    public static final int POSE_INPUT_W = 192;
    public static final int POSE_INPUT_H = 256;
    public static final int ACTION_INPUT_JOINTS = 13;
    public static final int ACTION_INPUT_CHANNELS = 2;

    // weights:
    public static final String OBJ_DET_WEIGHTS = "weights/objdet/nanodet_plus_m_320_coco_exam_iith_dmu_gen_data.all.ort";
    public static final String POSE_WEIGHTS = "weights/pose/pose_shufflenetv2_plus_pixel_shuffle.all.ort";
    public static final String ACTION_WEIGHTS = "weights/action/fc_net_2_channels_spine.disable.ort";

    // input names:
    public static final String OBJ_DET_INPUT_NAME = "data";
    public static final String POSE_INPUT_NAME = "images";
    public static final String ACTION_INPUT_NAME = "input";

    // object detection:
    public static final int OBJ_DET_NUM_CLS = 5;
    public static final int REG_MAX = 7;
    public static final float IOU_THRESH = 0.2f;
    public static final int[] STRIDES = {8, 16, 32, 64};
    public static final float[] SCORE_THRESH = {0.4f, 0.25f, 0.25f, 0.25f, 0.25f};
    public static final String[] OBJ_DET_CLASSES = {"person", "laptop", "mouse",
            "keyboard", "cell phone"};
    public static final int PERSON_CLS = 0;

    // action:
    public static final String[] ACTION_CLASSES = {
            "Hand reach out",
            "Look down",
            "Look outside",
            "Sitting"
    };
    public static final int SITTING_CLS = 3;
}
