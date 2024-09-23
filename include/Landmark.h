#ifndef CLASS_LANDMARK
#define CLASS_LANDMARK

#include <cstdint>
#include <map>
#include <memory>

#include "Pose.h"
#include "Camera.h"
#include "KeyFrame.h"

class KeyFrame;

class Landmark {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Landmark(const Eigen::Vector2d &px, KeyFrame *host, const std::shared_ptr<Camera> cam, 
        const uint64_t desc, const double z);
    Landmark() {}
    Eigen::Vector3d GetPcNorm() const;
    Eigen::Vector3d GetPc() const;
    Eigen::Vector3d GetPw() const;
    int Size() const; // 优化变量的维度
    void Update(const double delta_z, const bool useInvDepth);
    void UpdateUncertainty();
    bool Converge() const {return uncertainty_ < kConvergeDiff && z_ > kMinDepth && z_ < kMaxDepth;}
    bool Lost() {return host_ == nullptr && target_.empty();} // 地图点不再被更新
    std::vector<Eigen::Vector2d> FindMatches(const KeyFrame &kf2);

    double z_ = 1.0;
    double depthCov_ = std::pow(kMaxDepth, 2);
    double invZ_ = 1.0;
    double invDepthCov_ = std::pow(1./kMaxDepth, 2);


    double depthRange_[2] = {kMinDepth, kMaxDepth};
    double uncertainty_ = kMaxDepth;
    uint64_t descriptor_ = 0;

	//std::shared_ptr<KeyFrame> host_; 需确保host已经由智能指针管理，然后调用shared_from_this()来获取才行，不方便
    KeyFrame *host_; // cnchor frame
    Eigen::Vector2d uv_; // host帧下的像素坐标

	std::map<KeyFrame*, Eigen::Vector2d> target_;
    std::shared_ptr<Camera> cam_;
};

#endif
