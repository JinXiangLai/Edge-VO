#include "Landmark.h"

#include <cstdint>
#include "Pose.h"
#include "Utils.h"

using namespace std;
using namespace cv;

class KeyFrame;

Landmark::Landmark(const Eigen::Vector2i& px, KeyFrame* host,
                   const shared_ptr<Camera> cam, const uint64_t desc,
                   const double invZ)
    : invZ_(invZ), descriptor_(desc), host_(host), uv_(px), cam_(cam) {
    //invDepthCov_ = std::pow(1. / config->maxDepth, 2);
    depthRange_[0] = config->minDepth;
    depthRange_[1] = config->maxDepth;
}

Eigen::Vector3d Landmark::GetPcNorm() const {
    return cam_->InverseProject(uv_, 1.0);
}

Eigen::Vector3d Landmark::GetPc() const {
    return cam_->InverseProject(uv_, GetPositiveDepth(invZ_));
}

Eigen::Vector3d Landmark::GetPw() const {
    return host_->Twc_ * GetPc();
}

int Landmark::Size() const {
    return 1;
}

void Landmark::Update(const double delta_z, const bool useInvDepth) {
    double invZ = invZ_;
    if (useInvDepth) {
        invZ += delta_z;
    } else {
        const double z = 1.0 / invZ_ + delta_z;
        invZ = 1 / z;
    }

    // 为了保证优化算法的连续性，这里必须要修改
    //if (z > depthRange_[0] && z < depthRange_[1] || 1) {
    //    z_ = z;
    //    invZ_ = invZ;
    //    // TODO: 使用H*Δx = g，假设量测噪声为1个pixel，据此计算新的不确定度
    //    depthCov_ *= 0.9;
    //    UpdateUncertainty(false);
    //}
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
    invDepthCov_ = (cov1 * cov2) / sumCov;
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
    return (z < 0.5 || z > 10.0) && (obvTime_ > 5 || Converge());
}

bool Landmark::CheckInvDepthQualitySuccessByProject() {
    if (continousFailCheckNum_ > 3) {
        SetOutOfRange();
        return false;
    }

    if(continousPassCheckNum_ > 3) {
        passReprojectCheck_ = true;
        return true;
    }

    return false;
}

void Landmark::UpdateUncertainty(const bool updateObv) {
    const double stddev = sqrt(invDepthCov_);
    depthRange_[0] = GetPositiveDepth(invZ_ + stddev);
    depthRange_[1] = GetPositiveDepth(invZ_ - stddev);
    if (updateObv)
        ++obvTime_;
}

vector<Eigen::Vector2d> Landmark::FindMatches(const KeyFrame& kf2) {
    // TODO: 考虑不是host帧而是其他观测帧投影呢？
    const Pose T21 = kf2.Tcw_ * host_->Twc_;
    // 需要全局函数作用符"::"以实现类外全局函数的调用
    vector<Eigen::Vector2d> kp2 = ::FindMatches(*this, kf2, T21, *cam_);
    return kp2;
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
    if (invZ_ < 1e-9) {
        return false;
    }
    return sqrt(invDepthCov_) / invZ_ < 0.1 && obvTime_ > 5;
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
