#include "Landmark.h"

#include <cstdint>
#include "Pose.h"
#include "Utils.h"

using namespace std;
using namespace cv;

class KeyFrame;

Landmark::Landmark(const Eigen::Vector2i& px, KeyFrame* host,
                   const shared_ptr<Camera> cam, const uint64_t desc,
                   const double z)
    : invZ_(1.0 / z), descriptor_(desc), host_(host), uv_(px), cam_(cam) {
    invDepthCov_ = std::pow(1. / config->maxDepth, 2);
    depthRange_[0] = config->minDepth;
    depthRange_[1] = config->maxDepth;
}

Eigen::Vector3d Landmark::GetPcNorm() const {
    return cam_->InverseProject(uv_, 1.0);
}

Eigen::Vector3d Landmark::GetPc() const {
    return cam_->InverseProject(uv_, GetPositiveDepth(1.0 / invZ_));
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
    // 这个标准差是很不准的，所以不能用其判断
    // return uncertainty_ < config->maxDepthConvergeStd &&
    //     z_ > config->minDepth && z_ < config->maxDepth;
    // TODO：考虑把后续找不到匹配的深度估计值剔除才能最终实现一个基础版
    // OK，那些<0.5m深度的点，可以在一次观测收敛，但是被观测次数确实很少的，据此可以剔除
    const double stddev = sqrt(invDepthCov_);
    const double invZ1 = invZ_ - stddev;
    const double invZ2 = invZ_ + stddev;
    if (invZ1 < 1e-9 || invZ2 < 1e-9) {
        return false;
    }
    const double diff = abs(GetPositiveDepth(invZ1) - GetPositiveDepth(invZ2));
    return diff < 0.1;
    // return initFromPropagate_;
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
