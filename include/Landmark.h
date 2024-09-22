#ifndef CLASS_LANDMARK
#define CLASS_LANDMARK

#include "Pose.h"
#include "Camera.h"
#include "KeyFrame.h"

class KeyFrame;

class Landmark {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Landmark(const Eigen::Vector2d &px, std::shared_ptr<Pose> Twc, const std::shared_ptr<Camera> cam, 
        const double z = 1.0);
    Landmark() {}
    Eigen::Vector3d GetPcNorm() const;
    Eigen::Vector3d GetPc() const;
    Eigen::Vector3d GetPw() const;
    int Size() const; // 优化变量的维度
    void Update(const double delta_z, const bool useInvDepth);
    void UpdateUncertainty();
    bool Converge() const {return uncertainty_ < kConvergeDiff && z_ > kMinDepth && z_ < kMaxDepth;}

    Eigen::Vector2d uv_; // 像素坐标
    double z_ = 1.0;
    double depthCov_ = std::pow(kMaxDepth*0.5, 2);
    double invZ_ = 1.0;
    double invDepthCov_ = std::pow(2./kMaxDepth, 2);
    // anchor pose
    std::shared_ptr<Pose> Twc_;
    std::shared_ptr<Camera> cam_;
    double depthRange_[2] = {kMinDepth, kMaxDepth};
    double uncertainty_ = kMaxDepth * 0.5;
    uint64_t descriptor_ = 0;

	std::shared_ptr<KeyFrame> host_;
	std::vector<std::shared_ptr<KeyFrame> > target_;
};

#endif
