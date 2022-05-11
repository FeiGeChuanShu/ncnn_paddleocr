#include "common.h"
#include <jni.h>
#include <android/log.h>
bool cvPointCompare(const cv::Point& a, const cv::Point& b) {
    return a.x < b.x;
}
bool compareBoxWidth(const TextBox &a, const TextBox& b)
{
    return abs(a.boxPoint[0].x-a.boxPoint[1].x)>abs(b.boxPoint[0].x-b.boxPoint[1].x);
}

std::vector<cv::Point> getMinBoxes(const std::vector<cv::Point>& inVec, float& minSideLen, float& allEdgeSize) {
    std::vector<cv::Point> minBoxVec;
    cv::RotatedRect textRect = cv::minAreaRect(inVec);
    cv::Mat boxPoints2f;
    cv::boxPoints(textRect, boxPoints2f);

    float* p1 = (float*)boxPoints2f.data;
    std::vector<cv::Point> tmpVec;
    for (int i = 0; i < 4; ++i, p1 += 2) {
        tmpVec.emplace_back(int(p1[0]), int(p1[1]));
    }

    std::sort(tmpVec.begin(), tmpVec.end(), cvPointCompare);

    minBoxVec.clear();

    int index1, index2, index3, index4;
    if (tmpVec[1].y > tmpVec[0].y) {
        index1 = 0;
        index4 = 1;
    }
    else {
        index1 = 1;
        index4 = 0;
    }

    if (tmpVec[3].y > tmpVec[2].y) {
        index2 = 2;
        index3 = 3;
    }
    else {
        index2 = 3;
        index3 = 2;
    }

    minBoxVec.clear();

    minBoxVec.push_back(tmpVec[index1]);
    minBoxVec.push_back(tmpVec[index2]);
    minBoxVec.push_back(tmpVec[index3]);
    minBoxVec.push_back(tmpVec[index4]);

    minSideLen = (std::min)(textRect.size.width, textRect.size.height);
    allEdgeSize = 2.f * (textRect.size.width + textRect.size.height);

    return minBoxVec;
}

float boxScoreFast(const cv::Mat & inMat, const std::vector<cv::Point> & inBox) {
    std::vector<cv::Point> box = inBox;
    int width = inMat.cols;
    int height = inMat.rows;
    int maxX = -1, minX = 1000000, maxY = -1, minY = 1000000;
    for (int i = 0; i < box.size(); ++i) {
        if (maxX < box[i].x)
            maxX = box[i].x;
        if (minX > box[i].x)
            minX = box[i].x;
        if (maxY < box[i].y)
            maxY = box[i].y;
        if (minY > box[i].y)
            minY = box[i].y;
    }
    maxX = (std::min)((std::max)(maxX, 0), width - 1);
    minX = (std::max)((std::min)(minX, width - 1), 0);
    maxY = (std::min)((std::max)(maxY, 0), height - 1);
    minY = (std::max)((std::min)(minY, height - 1), 0);

    for (int i = 0; i < box.size(); ++i) {
        box[i].x = box[i].x - minX;
        box[i].y = box[i].y - minY;
    }

    std::vector<std::vector<cv::Point>> maskBox;
    maskBox.push_back(box);
    cv::Mat maskMat(maxY - minY + 1, maxX - minX + 1, CV_8UC1, cv::Scalar(0, 0, 0));
    cv::fillPoly(maskMat, maskBox, cv::Scalar(1, 1, 1), 1);
    return cv::mean(inMat(cv::Rect(cv::Point(minX, minY), cv::Point(maxX + 1, maxY + 1))).clone(),
        maskMat).val[0];
}

std::vector<cv::Point> unClip(const std::vector<cv::Point> & inBox, float perimeter, float unClipRatio) {
    std::vector<cv::Point> outBox;
    ClipperLib::Path poly;

    for (int i = 0; i < inBox.size(); ++i) {
        poly.push_back(ClipperLib::IntPoint(inBox[i].x, inBox[i].y));
    }

    double distance = unClipRatio * ClipperLib::Area(poly) / (double)perimeter;

    ClipperLib::ClipperOffset clipperOffset;
    clipperOffset.AddPath(poly, ClipperLib::JoinType::jtRound, ClipperLib::EndType::etClosedPolygon);
    ClipperLib::Paths polys;
    polys.push_back(poly);
    clipperOffset.Execute(polys, distance);

    outBox.clear();
    std::vector<cv::Point> rsVec;
    for (int i = 0; i < polys.size(); ++i) {
        ClipperLib::Path tmpPoly = polys[i];
        for (int j = 0; j < tmpPoly.size(); ++j) {
            outBox.emplace_back(tmpPoly[j].X, tmpPoly[j].Y);
        }
    }
    return outBox;
}

cv::Mat getRotateCropImage(const cv::Mat& src, std::vector<cv::Point> box) {
    cv::Mat image;
    src.copyTo(image);
    std::vector<cv::Point> points = box;

    int collectX[4] = { box[0].x, box[1].x, box[2].x, box[3].x };
    int collectY[4] = { box[0].y, box[1].y, box[2].y, box[3].y };
    int left = int(*std::min_element(collectX, collectX + 4));
    int right = int(*std::max_element(collectX, collectX + 4));
    int top = int(*std::min_element(collectY, collectY + 4));
    int bottom = int(*std::max_element(collectY, collectY + 4));

    cv::Mat imgCrop;
    image(cv::Rect(left, top, right - left, bottom - top)).copyTo(imgCrop);

    for (int i = 0; i < points.size(); i++) {
        points[i].x -= left;
        points[i].y -= top;
    }


    int imgCropWidth = int(sqrt(pow(points[0].x - points[1].x, 2) +
        pow(points[0].y - points[1].y, 2)));
    int imgCropHeight = int(sqrt(pow(points[0].x - points[3].x, 2) +
        pow(points[0].y - points[3].y, 2)));

    cv::Point2f ptsDst[4];
    ptsDst[0] = cv::Point2f(0., 0.);
    ptsDst[1] = cv::Point2f(imgCropWidth, 0.);
    ptsDst[2] = cv::Point2f(imgCropWidth, imgCropHeight);
    ptsDst[3] = cv::Point2f(0.f, imgCropHeight);

    cv::Point2f ptsSrc[4];
    ptsSrc[0] = cv::Point2f(points[0].x, points[0].y);
    ptsSrc[1] = cv::Point2f(points[1].x, points[1].y);
    ptsSrc[2] = cv::Point2f(points[2].x, points[2].y);
    ptsSrc[3] = cv::Point2f(points[3].x, points[3].y);

    cv::Mat M = cv::getPerspectiveTransform(ptsSrc, ptsDst);

    cv::Mat partImg;
    cv::warpPerspective(imgCrop, partImg, M,
        cv::Size(imgCropWidth, imgCropHeight),
        cv::BORDER_REPLICATE);

    if (float(partImg.rows) >= float(partImg.cols) * 1.5) {
        cv::Mat srcCopy = cv::Mat(partImg.rows, partImg.cols, partImg.depth());
        cv::transpose(partImg, srcCopy);
        cv::flip(srcCopy, srcCopy, 0);
        return srcCopy;
    }
    else {
        return partImg;
    }
}
std::vector<cv::Mat> getPartImages(const cv::Mat& src, std::vector<TextBox>& textBoxes)
{
    std::sort(textBoxes.begin(),textBoxes.end(),compareBoxWidth);
    std::vector<cv::Mat> partImages;
    if(textBoxes.size() > 0)
    {
        for (int i = 0; i < textBoxes.size(); ++i)
        {
            cv::Mat partImg = getRotateCropImage(src, textBoxes[i].boxPoint);
            partImages.emplace_back(partImg);
        }
    }

    return partImages;
}
cv::Mat matRotateClockWise180(cv::Mat src) {
    flip(src, src, 0);
    flip(src, src, 1);
    return src;
}

cv::Mat makePadding(cv::Mat& src, const int padding) {
    if (padding <= 0) return src;
    cv::Scalar paddingScalar = { 255, 255, 255 };
    cv::Mat paddingSrc;
    cv::copyMakeBorder(src, paddingSrc, padding, padding, padding, padding, cv::BORDER_ISOLATED, paddingScalar);
    return paddingSrc;
}