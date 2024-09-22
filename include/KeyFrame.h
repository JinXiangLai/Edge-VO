#ifndef CLASS_KEYFRAME
#define CLASS_KEYFRAME

#include <memory>

#include "Camera.h"
#include "Pose.h"
#include "Landmark.h"

class Landmark;

struct TupleHash {
    size_t operator()(const std::tuple<int, int> &v) const{
        const int v1 = std::get<0>(v);
        const int v2 = std::get<1>(v);
        return (v1 << 1) + (v2 >> 1);
    }
};

class KeyFrame {
public:
    KeyFrame(const cv::Mat &img, const Pose &Twc, std::shared_ptr<Camera> cam, const int level = 1)
    : grayImg_(img)
    , cam_(cam)
    , Twc_ {Twc}
    , level_(level) {
        edgeImg_.resize(level);
        dist_.resize(level);
        dx_.resize(level);
        dy_.resize(level);
        unPx_.resize(level);
    }
    // 可能需要corase2fine的配准
    void CannyEdgeDetect();
    void GenerateDTandDerivative();
    std::vector<Eigen::Vector2d> FindMatches(const Eigen::Vector2d &kp1, const Pose &Twc1);
    size_t GenerateLandmark(KeyFrame &kf1, std::vector<std::vector<Eigen::Vector2d> > &debugGoodKp1, 
        std::vector<std::vector<Eigen::Vector2d> >&debugGoodKp2, const int equalparts);
    void UpdateDepth(const KeyFrame &kf2);
    cv::Mat grayImg_;
    // canny边缘图像已经去畸变了
    std::vector<cv::Mat> edgeImg_, dist_, dx_, dy_;
    std::shared_ptr<Camera> cam_;
    std::vector<Landmark> landmark_;
    Pose Twc_;
    int level_ = 1;
    std::vector<std::vector<Eigen::Vector2d> > unPx_; // 像素平面上的去畸变点
    // std::vector<Eigen::Matrix<float, kDescriptorPatchSize, 1> > descriptor_;
    std::vector<uint64_t> descriptor_; 
    static constexpr int descDim = 63;
    std::unordered_map<std::tuple<int, int>, int, TupleHash> pointMapId_; // 像素坐标与索引的映射
};

#endif
