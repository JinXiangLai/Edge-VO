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
    // EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    KeyFrame(const cv::Mat &img, const Pose &Twc, std::shared_ptr<Camera> cam, const int id, const int level = 1);
    ~KeyFrame();
    KeyFrame(){}
    KeyFrame(const KeyFrame &f);
    void operator =(const KeyFrame &f);

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
    void Update(const Eigen::Vector3d &delta_q, const Eigen::Vector3d &delta_t);
    void SetTwc(const Pose &Twc);
    int TrackLandmarkByEpilorLine(const KeyFrame &kf1);
    double CullingBadDepth(KeyFrame *kf2);
    double CalculateSSD(double *v1, double *v2, double avg1, double avg2, const int desLen);
    std::vector<double> CalculateDescriptor(const cv::Mat &grayImg, const Eigen::Vector2d &px, const Eigen::Vector2d &epNorm, const int len=5);
    void ReleaseMat();
    void FuseDepth();

    std::vector<Eigen::Vector2d> FindMatchesWithEpipolarConstraintOnImagePlane(const KeyFrame* kf2, Landmark* lk1, Eigen::Vector2d &deltaPx2);

    unsigned int id_;
    cv::Mat grayImg_, debugGrayImg_;
    // canny边缘图像已经去畸变了
    std::vector<cv::Mat> edgeImg_, dist_, dx_, dy_;
    std::shared_ptr<Camera> cam_;
    Pose Twc_;
    // TODO： 增加该字段，减小Inverse()次数
    Pose Tcw_;
    Pose priorTwc_;
    int level_ = 1;
    std::vector<std::vector<Eigen::Vector2d> > unPx_; // 像素平面上的去畸变点
    std::vector<Landmark* > landmark_; // 成员变量内存在指针，需要手写拷贝构造函数
    // std::vector<Eigen::Matrix<float, kDescriptorPatchSize, 1> > descriptor_;
    std::vector<uint64_t> descriptor_; 
    static constexpr int descDim = 63;
    std::unordered_map<Eigen::Vector2i, int, TupleHash> pointMapId_; // 像素坐标与vector索引的映射
    bool outOfRange_ = false;
    int convergeEdgeNum_ = 0;
};

#endif
