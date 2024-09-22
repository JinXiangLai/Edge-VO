#include "Landmark.h"

using namespace std;
using namespace cv;

class KeyFrame;

Landmark::Landmark(const Eigen::Vector2d &px, shared_ptr<Pose> Twc, const shared_ptr<Camera> cam, 
    const double z)
    : z_(z)
    , invZ_(1.0/z)
    , uv_(px)
    , Twc_(Twc)
    , cam_(cam) {}

Eigen::Vector3d Landmark::GetPcNorm() const {
    return cam_->InverseProject(uv_.cast<int>(), 1.0);
}

Eigen::Vector3d Landmark::GetPc() const {
    return cam_->InverseProject(uv_.cast<int>(), z_);
}

Eigen::Vector3d Landmark::GetPw() const {
    return (*Twc_) * GetPc();
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
}

void Landmark::UpdateUncertainty() {
    uncertainty_ = sqrt(depthCov_);
    invZ_ = 1.0 / z_;
    depthRange_[0] = max(kMinDepth, z_ - 3 * uncertainty_);
    depthRange_[1] = min(kMaxDepth, z_ + 3 * uncertainty_);
}
