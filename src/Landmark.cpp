#include "Landmark.h"

#include <cstdint>
#include "Pose.h"
#include "Utils.h"

using namespace std;
using namespace cv;

class KeyFrame;

Landmark::Landmark(const Eigen::Vector2d& px, KeyFrame* host,
                   const shared_ptr<Camera> cam, const uint64_t desc,
                   const double invZ)
    : invZ_(invZ), descriptor_(desc), host_(host), uv_(px), cam_(cam) {}

Eigen::Vector3d Landmark::GetPcNorm() const {
    return cam_->InverseProject(uv_, 1.0);
}

Eigen::Vector3d Landmark::GetPc(const bool useBackUpStatus) const {
    if (!useBackUpStatus)
        return cam_->InverseProject(uv_, GetPositiveDepth(invZ_));

    return cam_->InverseProject(uv_, GetPositiveDepth(invZback_));
}

Eigen::Vector3d Landmark::GetPw(const bool useBackUpStatus) const {
    return host_->Twc_ * GetPc(useBackUpStatus);
}

int Landmark::Size() const {
    return 1;
}

void Landmark::Update(const double delta_z) {
    invZ_ += delta_z;
}

void Landmark::SetTriangulateResult(const double invZ) {
    invZ_ = invZ;
    obvTime_ = 1;
    initialized_ = true;
}

bool Landmark::ObvUpdate(const double invDepth, const double variance) {
    const double diff = abs(invZ_ - invDepth);
    if (diff > sqrt(invDepthCov_) * 2.0) {
        invDepthCov_ *= expandRatio;
        failObvTime_++;
        if (failObvTime_ > 2 * obvTime_) {
            SetOutOfRange();
        }
        return false;
    }

    const double &u2 = invDepth, &cov2 = variance;  // 考虑基线的影响
    const double &u1 = invZ_, &cov1 = invDepthCov_;
    const double sumCov = cov1 + cov2;
    invZ_ = (u2 * cov1 + u1 * cov2) / sumCov;
    invDepthCov_ = std::max(0.001, (cov1 * cov2) / sumCov);
    //UpdateUncertainty(true);
    obvTime_++;
    return true;
}

bool Landmark::FuseInvDepth(const Landmark& lk2) {
    const double &u1 = invZ_, &u2 = lk2.invZ_;
    const double &cov1 = invDepthCov_, &cov2 = lk2.invDepthCov_;
    if (pow(u1 - u2, 2) > 4 * cov1) {
        return false;
    }
    const double sumCov = cov1 + cov2;
    invZ_ = (u1 * cov2 + u2 * cov1) / sumCov;
    // invDepthCov_ = (cov1 * cov2) / sumCov;
    return true;
}

bool Landmark::AbnormalConvergeLandmark() {
    // 用于debug输出异常的landmark以优化匹配算法
    const double z = GetPositiveDepth(invZ_);
    return (z < 0.5 || z > 10.0) && Converge();
}

bool Landmark::CheckInvDepthQualitySuccessByProject() {
    if (continousFailCheckNum_ > 3) {
        SetOutOfRange();
        return false;
    }

    if (continousPassCheckNum_ > 3) {
        passReprojectCheck_ = true;
        return true;
    }

    return false;
}

bool Landmark::TransformHost2OtherKF(KeyFrame* kf2) {
    const Eigen::Vector3d pc1 = GetPc();
    const Pose T21 = kf2->Tcw_ * host_->Twc_;
    const Eigen::Vector3d pc2 = T21 * pc1;
    if (pc2.z() < kMinSceneDepthInCamera) {
        return false;
    }
    invZ_ = 1.0 / pc2.z();
    if (target_.count(host_)) {
        target_.erase(host_);
    }
    host_ = kf2;
    uv_ = target_.at(kf2);
    // TODO：暂不使用首次雅可比
    return true;
}

void Landmark::CopyStatus() {
    invZback_ = invZ_;
}

void Landmark::BackUpStatus() {
    invZ_ = invZback_;
}

void Landmark::ResetFEJ() {
    J_Pc2_Twc2.clear();
    J_Pc2_Pw.clear();
    J_Pw_z.clear();
    J_Pw_Twc1.clear();
    J_Pc2_T12_.clear();
    noUsed_ = false;
    matchNextPixel_.setZero();
}

bool Landmark::Converge() const {
    //if (invZ_ < 1e-9) {
    //    return false;
    //}
    //const double stddev = sqrt(invDepthCov_);
    //return (stddev < 0.001 || stddev / invZ_ < 0.1) && obvTime_ > 5;
    return obvTime_ > 0;
}

bool Landmark::ManySupport() const {

#if 1
    return initFromPropagate_ || 1;
#else
    const int w = host_->grayImg_.cols, h = host_->grayImg_.rows;
    constexpr int minNearSupport = 5;
    int nearSupport = 0;
    const int x = uv_.x(), y = uv_.y();
    for (int i = -1; i < 2; ++i) {
        for (int j = -1; j < 2; ++j) {
            Eigen::Vector2i px{x + j, y + i};
#if USE_POINT_MAP_ID
            if (host_->pointMapId_.count(px)) {
                const int id = host_->pointMapId_.at(px);
#else
            const int id = px.y() * w + px.x();
            if (host_->pointMapId_.count(id)) {
                const int id = host_->pointMapId_.at(id);
#endif
                Landmark* lk = host_->landmark_[id];
                if (lk == nullptr || lk->IsOutOfRange()) {
                    continue;
                }
                const double diff = abs(z_ - lk->z_);
                if (diff < 2 * uncertainty_) {
                    ++nearSupport;
                }
            }
        }
    }
    return nearSupport > minNearSupport;
#endif
}
