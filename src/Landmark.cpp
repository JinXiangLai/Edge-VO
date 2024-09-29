#include "Landmark.h"

#include "Pose.h"
#include "Utils.h"
#include <cstdint>

using namespace std;
using namespace cv;

class KeyFrame;

Landmark::Landmark(const Eigen::Vector2d &px, KeyFrame *host, const shared_ptr<Camera> cam, 
    const uint64_t desc, const double z)
    : z_(z)
    , invZ_(1.0/z)
    , descriptor_(desc)
    , host_(host)
    , uv_(px)
    , cam_(cam){
        depthCov_ = std::pow(config->maxDepth, 2);
        invDepthCov_ = std::pow(1./config->maxDepth, 2);
        depthRange_[0] = config->minDepth;
        depthRange_[1] = config->maxDepth;
        uncertainty_ = config->maxDepth;
    }

Eigen::Vector3d Landmark::GetPcNorm() const {
    return cam_->InverseProject(uv_.cast<int>(), 1.0);
}

Eigen::Vector3d Landmark::GetPc() const {
    return cam_->InverseProject(uv_.cast<int>(), z_);
}

Eigen::Vector3d Landmark::GetPw() const {
    return host_->Twc_ * GetPc();
}

int Landmark::Size() const {return 1;}

void Landmark::Update(const double delta_z, const bool useInvDepth) {
    if(useInvDepth) {
        invZ_ += delta_z;
        z_ = 1/invZ_;
    } else {
        z_ += delta_z; 
        invZ_ = 1/z_;
    }

    // TODO: 使用H*Δx = g，假设量测噪声为1个pixel，据此计算新的不确定度
    depthRange_[0] = max(config->minDepth, z_ - 2*uncertainty_);
    depthRange_[1] = min(config->maxDepth, z_ + 2*uncertainty_);
}

void Landmark::UpdateUncertainty() {
    uncertainty_ = sqrt(depthCov_);
    invZ_ = 1.0 / z_;
    depthRange_[0] = max(config->minDepth, z_ - 3 * uncertainty_);
    depthRange_[1] = min(config->maxDepth, z_ + 3 * uncertainty_);
}

vector<Eigen::Vector2d> Landmark::FindMatches(const KeyFrame &kf2) {
    // TODO: 考虑不是host帧而是其他观测帧投影呢？
    const Pose T21 = kf2.Tcw_ * host_->Twc_;
    // 需要全局函数作用符"::"以实现类外全局函数的调用
    vector<Eigen::Vector2d> kp2 = ::FindMatches(*this, kf2, T21, *cam_);
    return kp2;
}

