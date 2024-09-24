#ifndef CLASS_KEYFRAME
#define CLASS_KEYFRAME

#include <memory>

#include "Camera.h"
#include "Pose.h"
#include "Landmark.h"

class Landmark;

struct TupleHash {
    size_t operator()(const Eigen::Vector2i &v) const{
        const int v1 = v[0];
        const int v2 = v[1];
        return (v1 << 1) + (v2 >> 1);
    }
};

class KeyFrame {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    KeyFrame(const cv::Mat &img, const Pose &Twc, std::shared_ptr<Camera> cam, const int level = 1)
    : grayImg_(img)
    , cam_(cam)
    , Twc_ {Twc}
    , priorTwc_(Twc)
    , level_(level) {
        edgeImg_.resize(level);
        dist_.resize(level);
        dx_.resize(level);
        dy_.resize(level);
        unPx_.resize(level);
    }
    ~KeyFrame() {std::cout << "delete keyframe: " << this << std::endl;}
    // 可能需要corase2fine的配准
    void CannyEdgeDetect();
    void GenerateDTandDerivative();
    size_t GenerateLandmark(KeyFrame &kf1, std::vector<std::vector<Eigen::Vector2d> > &debugGoodKp1, 
        std::vector<std::vector<Eigen::Vector2d> >&debugGoodKp2, const int equalparts);
    size_t InitializeLandmark();
    int ReuseLandmark(KeyFrame *kf1);
    double UpdateDepth(const KeyFrame &kf2);
    void SetOutOfRange() {outOfRange_ = true;}
    bool IsOutOfRange() const {return outOfRange_;}
    cv::Mat grayImg_;
    // canny边缘图像已经去畸变了
    std::vector<cv::Mat> edgeImg_, dist_, dx_, dy_;
    std::shared_ptr<Camera> cam_;
    Pose Twc_;
    const Pose priorTwc_;
    int level_ = 1;
    std::vector<std::vector<Eigen::Vector2d> > unPx_; // 像素平面上的去畸变点
    std::vector<Landmark* > landmark_;
    // std::vector<Eigen::Matrix<float, kDescriptorPatchSize, 1> > descriptor_;
    std::vector<uint64_t> descriptor_; 
    static constexpr int descDim = 63;
    std::unordered_map<Eigen::Vector2i, int, TupleHash> pointMapId_; // 像素坐标与vector索引的映射
    bool outOfRange_ = false;
};

#endif
