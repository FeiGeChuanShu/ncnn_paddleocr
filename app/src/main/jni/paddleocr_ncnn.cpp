// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <android/log.h>

#include <jni.h>

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
// ncnn
#include "layer.h"
#include "net.h"
#include "benchmark.h"
#include "common.h"
static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;
const int dstHeight = 32;//when use PP-OCRv3 it should be 48
ncnn::Net dbNet;
ncnn::Net crnnNet;

std::vector<std::string> keys;
char *readKeysFromAssets(AAssetManager *mgr)
{
    if (mgr == NULL) {
        return NULL;
    }
    char *buffer;

    AAsset *asset = AAssetManager_open(mgr, "paddleocr_keys.txt", AASSET_MODE_UNKNOWN);
    if (asset == NULL) {
        return NULL;
    }

    off_t bufferSize = AAsset_getLength(asset);
    buffer = (char *) malloc(bufferSize + 1);
    buffer[bufferSize] = 0;
    int numBytesRead = AAsset_read(asset, buffer, bufferSize);
    AAsset_close(asset);

    return buffer;
}


std::vector<TextBox> findRsBoxes(const cv::Mat& fMapMat, const cv::Mat& norfMapMat,
    const float boxScoreThresh, const float unClipRatio) 
{
    float minArea = 3;
    std::vector<TextBox> rsBoxes;
    rsBoxes.clear();
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(norfMapMat, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    for (int i = 0; i < contours.size(); ++i) 
    {
        float minSideLen, perimeter;
        std::vector<cv::Point> minBox = getMinBoxes(contours[i], minSideLen, perimeter);
        if (minSideLen < minArea)
            continue;
        float score = boxScoreFast(fMapMat, contours[i]);
        if (score < boxScoreThresh)
            continue;
        //---use clipper start---
        std::vector<cv::Point> clipBox = unClip(minBox, perimeter, unClipRatio);
        std::vector<cv::Point> clipMinBox = getMinBoxes(clipBox, minSideLen, perimeter);
        //---use clipper end---

        if (minSideLen < minArea + 2)
            continue;

        for (int j = 0; j < clipMinBox.size(); ++j) 
        {
            clipMinBox[j].x = (clipMinBox[j].x / 1.0);
            clipMinBox[j].x = (std::min)((std::max)(clipMinBox[j].x, 0), norfMapMat.cols);

            clipMinBox[j].y = (clipMinBox[j].y / 1.0);
            clipMinBox[j].y = (std::min)((std::max)(clipMinBox[j].y, 0), norfMapMat.rows);
        }
        
        rsBoxes.emplace_back(TextBox{ clipMinBox, score });
    }
    reverse(rsBoxes.begin(), rsBoxes.end());

    return rsBoxes;
}

std::vector<TextBox> getTextBoxes(const cv::Mat & src, float boxScoreThresh, float boxThresh, float unClipRatio)
{
    int width = src.cols;
    int height = src.rows;
    int target_size = 640;
    // pad to multiple of 32
    int w = width;
    int h = height;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat input = ncnn::Mat::from_pixels_resize(src.data, ncnn::Mat::PIXEL_RGB, width, height, w, h);

    // pad to target_size rectangle
    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(input, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);

    const float meanValues[3] = { 0.485 * 255, 0.456 * 255, 0.406 * 255 };
    const float normValues[3] = { 1.0 / 0.229 / 255.0, 1.0 / 0.224 / 255.0, 1.0 / 0.225 / 255.0 };

    in_pad.substract_mean_normalize(meanValues, normValues);
    ncnn::Extractor extractor = dbNet.create_extractor();

    extractor.input("input0", in_pad);
    ncnn::Mat out;
    extractor.extract("out1", out);

    cv::Mat fMapMat(in_pad.h, in_pad.w, CV_32FC1, (float*)out.data);
    cv::Mat norfMapMat;
    norfMapMat = fMapMat > boxThresh;

    cv::dilate(norfMapMat, norfMapMat, cv::Mat(), cv::Point(-1, -1), 1);

    std::vector<TextBox> result = findRsBoxes(fMapMat, norfMapMat, boxScoreThresh, 2.0f);
    for(int i = 0; i < result.size(); i++)
    {
        for(int j = 0; j < result[i].boxPoint.size(); j++)
        {
            float x = (result[i].boxPoint[j].x-(wpad/2))/scale;
            float y = (result[i].boxPoint[j].y-(hpad/2))/scale;
            x = std::max(std::min(x,(float)(width-1)),0.f);
            y = std::max(std::min(y,(float)(height-1)),0.f);
            result[i].boxPoint[j].x = x;
            result[i].boxPoint[j].y = y;
        }
    }

    return result;
}

template<class ForwardIterator>
inline static size_t argmax(ForwardIterator first, ForwardIterator last) {
    return std::distance(first, std::max_element(first, last));
}

TextLine scoreToTextLine(const std::vector<float>& outputData, int h, int w)
{
    int keySize = keys.size();
    std::string strRes;
    std::vector<float> scores;
    int lastIndex = 0;
    int maxIndex;
    float maxValue;

    for (int i = 0; i < h; i++)
    {
        maxIndex = 0;
        maxValue = -1000.f;

        maxIndex = int(argmax(outputData.begin()+i*w, outputData.begin()+i*w+w));
        maxValue = float(*std::max_element(outputData.begin()+i*w, outputData.begin()+i*w+w));// / partition;
        if (maxIndex > 0 && maxIndex < keySize && (!(i > 0 && maxIndex == lastIndex))) {
            scores.emplace_back(maxValue);
            strRes.append(keys[maxIndex - 1]);
        }
        lastIndex = maxIndex;
    }
    return { strRes, scores };
}

TextLine getTextLine(const cv::Mat & src)
{
    float scale = (float)dstHeight / (float)src.rows;
    int dstWidth = int((float)src.cols * scale);

    cv::Mat srcResize;
    cv::resize(src, srcResize, cv::Size(dstWidth, dstHeight));
    //if you use PP-OCRv3 you should change PIXEL_RGB to PIXEL_RGB2BGR
    ncnn::Mat input = ncnn::Mat::from_pixels(srcResize.data, ncnn::Mat::PIXEL_RGB,srcResize.cols, srcResize.rows);
    const float mean_vals[3] = { 127.5, 127.5, 127.5 };
    const float norm_vals[3] = { 1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5 };
    input.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor extractor = crnnNet.create_extractor();
    //extractor.set_num_threads(2);
    extractor.input("input", input);

    ncnn::Mat out;
    extractor.extract("out", out);
    float* floatArray = (float*)out.data;
    std::vector<float> outputData(floatArray, floatArray + out.h * out.w);
    TextLine res = scoreToTextLine(outputData, out.h, out.w);
    return res;
}

std::vector<TextLine> getTextLines(std::vector<cv::Mat> & partImg) {
    int size = partImg.size();
    std::vector<TextLine> textLines(size);
    for (int i = 0; i < size; ++i)
    {
        TextLine textLine = getTextLine(partImg[i]);
        textLines[i] = textLine;
    }
    return textLines;
}

extern "C" {

// FIXME DeleteGlobalRef is missing for objCls
static jclass objCls = NULL;
static jmethodID constructortorId;
static jfieldID x0Id;
static jfieldID y0Id;
static jfieldID x1Id;
static jfieldID y1Id;
static jfieldID x2Id;
static jfieldID y2Id;
static jfieldID x3Id;
static jfieldID y3Id;
static jfieldID labelId;
static jfieldID probId;

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "PaddleOCRNcnn", "JNI_OnLoad");

    ncnn::create_gpu_instance();

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "PaddleOCRNcnn", "JNI_OnUnload");

    ncnn::destroy_gpu_instance();
}

// public native boolean Init(AssetManager mgr);
JNIEXPORT jboolean JNICALL Java_com_tencent_paddleocrncnn_PaddleOCRNcnn_Init(JNIEnv* env, jobject thiz, jobject assetManager)
{
    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = 4;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;
    opt.use_packing_layout = true;

    // use vulkan compute
    if (ncnn::get_gpu_count() != 0)
        opt.use_vulkan_compute = true;

    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    dbNet.opt = opt;
    crnnNet.opt = opt;

    // init param
    {
        int ret = dbNet.load_param(mgr, "pdocrv2.0_det-op.param");
        if (ret != 0)
        {
            __android_log_print(ANDROID_LOG_WARN, "PaddleocrNcnn", "load_dbNet_param failed");
            return JNI_FALSE;
        }
        
        ret = crnnNet.load_param(mgr, "pdocrv2.0_rec-op.param");
        
        if (ret != 0)
        {
            __android_log_print(ANDROID_LOG_WARN, "PaddleocrNcnn", "load_crnnNet_param failed");
            return JNI_FALSE;
        }
    }

    // init bin
    {
        int ret = dbNet.load_model(mgr, "pdocrv2.0_det-op.bin");
        if (ret != 0)
        {
            __android_log_print(ANDROID_LOG_WARN, "PaddleocrNcnn", "load_dbNet_model failed");
            return JNI_FALSE;
        }
        
        ret = crnnNet.load_model(mgr, "pdocrv2.0_rec-op.bin");
        
        if (ret != 0)
        {
            __android_log_print(ANDROID_LOG_WARN, "PaddleocrNcnn", "load_crnnNet_model failed");
            return JNI_FALSE;
        }
    }
    
    //load keys
    char *buffer = readKeysFromAssets(mgr);
    if (buffer != NULL) {
        std::istringstream inStr(buffer);
        std::string line;
        int size = 0;
        while (getline(inStr, line)) {
            keys.emplace_back(line);
            size++;
        }
        free(buffer);
    } else {
        return false;
    }
    
    
    // init jni glue
    jclass localObjCls = env->FindClass("com/tencent/paddleocrncnn/PaddleOCRNcnn$Obj");
    objCls = reinterpret_cast<jclass>(env->NewGlobalRef(localObjCls));

    constructortorId = env->GetMethodID(objCls, "<init>", "(Lcom/tencent/paddleocrncnn/PaddleOCRNcnn;)V");

    x0Id = env->GetFieldID(objCls, "x0", "F");
    y0Id = env->GetFieldID(objCls, "y0", "F");
    x1Id = env->GetFieldID(objCls, "x1", "F");
    y1Id = env->GetFieldID(objCls, "y1", "F");
    x2Id = env->GetFieldID(objCls, "x2", "F");
    y2Id = env->GetFieldID(objCls, "y2", "F");
    x3Id = env->GetFieldID(objCls, "x3", "F");
    y3Id = env->GetFieldID(objCls, "y3", "F");
    labelId = env->GetFieldID(objCls, "label", "Ljava/lang/String;");
    probId = env->GetFieldID(objCls, "prob", "F");

    return JNI_TRUE;
}

// public native Obj[] Detect(Bitmap bitmap, boolean use_gpu);
JNIEXPORT jobjectArray JNICALL Java_com_tencent_paddleocrncnn_PaddleOCRNcnn_Detect(JNIEnv* env, jobject thiz, jobject bitmap, jboolean use_gpu)
{
    if (use_gpu == JNI_TRUE && ncnn::get_gpu_count() == 0)
    {
        return NULL;
        //return env->NewStringUTF("no vulkan capable gpu");
    }

    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap, &info);
    const int width = info.width;
    const int height = info.height;
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
        return NULL;

    ncnn::Mat in = ncnn::Mat::from_android_bitmap(env, bitmap, ncnn::Mat::PIXEL_RGB);

    cv::Mat rgb = cv::Mat::zeros(in.h,in.w,CV_8UC3);
    in.to_pixels(rgb.data, ncnn::Mat::PIXEL_RGB);

    std::vector<TextBox> objects; 
    objects = getTextBoxes(rgb, 0.4, 0.3, 2.0);

    std::vector<cv::Mat> partImages = getPartImages(rgb, objects);
    std::vector<TextLine> textLines = getTextLines(partImages);

    if(textLines.size() > 0)
    {
        for(int i = 0; i < textLines.size(); i++)
            objects[i].text = textLines[i].text;
    }
    // objects to Obj[]
    jobjectArray jObjArray = env->NewObjectArray(objects.size(), objCls, NULL);

    for (size_t i=0; i<objects.size(); i++)
    {
        jobject jObj = env->NewObject(objCls, constructortorId, thiz);

        float x0 = objects[i].boxPoint[0].x;
        float y0 = objects[i].boxPoint[0].y;
        float x1 = objects[i].boxPoint[1].x;
        float y1 = objects[i].boxPoint[1].y;
        float x2 = objects[i].boxPoint[2].x;
        float y2 = objects[i].boxPoint[2].y;
        float x3 = objects[i].boxPoint[3].x;
        float y3 = objects[i].boxPoint[3].y;

        env->SetFloatField(jObj, x0Id, x0);
        env->SetFloatField(jObj, y0Id, y0);
        env->SetFloatField(jObj, x1Id, x1);
        env->SetFloatField(jObj, y1Id, y1);
        env->SetFloatField(jObj, x2Id, x2);
        env->SetFloatField(jObj, y2Id, y2);
        env->SetFloatField(jObj, x3Id, x3);
        env->SetFloatField(jObj, y3Id, y3);
        env->SetObjectField(jObj, labelId, env->NewStringUTF(objects[i].text.c_str()));
        env->SetFloatField(jObj, probId, objects[i].score);

        env->SetObjectArrayElement(jObjArray, i, jObj);
    }

    return jObjArray;
}

}
