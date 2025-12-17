#ifndef OPTICAL_FLOW_STRUCT
#define OPTICAL_FLOW_STRUCT

#include <vector>

#include <opencv2/opencv.hpp>
#include "Landmark.h"

class Landmark;

struct OpticalFlowStruct {
    cv::Mat prevImg_;
    std::vector<cv::Point2f> prevPts_;
    std::vector<std::shared_ptr<Landmark>> trackLandmark_;
    size_t totalFeatureCreated_ = 0;
    size_t historyLandmarkNum_ = 0;  // 记录历史跟踪点的数量以区分上一KF的
    double meanParallax_ = 0.;
    int usefulParallaxNum_ = 0;
    size_t GetTrackFeatureNum() { return prevPts_.size(); }
    double GetTrackFeatureRatio() {
        return double(GetTrackFeatureNum()) / totalFeatureCreated_;
    }
    double GetHistoryTrackFeatureRatio() {
        return double(historyLandmarkNum_) / totalFeatureCreated_;
    }
    size_t SetTotalFeatureCreated() {
        totalFeatureCreated_ = GetTrackFeatureNum();
        return totalFeatureCreated_;
    }
    void Reset() {
        prevImg_.release();
        prevPts_.clear();
        trackLandmark_.clear();
        totalFeatureCreated_ = 0;
        historyLandmarkNum_ = 0;
    }
    void Set(const cv::Mat& img, const std::vector<cv::Point2f>& prevPts,
             const std::vector<std::shared_ptr<Landmark>>& landmark,
             const int totalFeatureCreated, const int historyLandmarkNum);
    void RemoveUselessLandmark();

    
};

// 全局变量声明
extern std::mutex globalOptFlwMutex;
extern OpticalFlowStruct globalOptFlw;

#endif