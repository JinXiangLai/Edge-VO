#include "Landmark.h"

#include <cstdint>
#include "Pose.h"
#include "Utils.h"

using namespace std;
using namespace cv;

std::shared_ptr<Camera> Landmark::cam_;

class KeyFrame;

Landmark::Landmark(const int kpRow, KeyFrame* host,
                   const shared_ptr<Camera> cam, const double invZ)
    : invZ_(invZ), host_(host), kpRow_(kpRow) {
    if (cam_ == nullptr) {
        cam_ = cam;
    }
    // 添加与其初始化帧的相互观测
    AddNewKFobservation(host_, kpRow);
}

Eigen::Vector3d Landmark::GetPcNorm() const {
    return cam_->InverseProject(GetHostFrameObv(), 1.0);
}

Eigen::Vector3d Landmark::GetPc(const bool useBackUpStatus) const {
    if (cam_ == nullptr) {
        cout << this << " cam_ is nullptr!" << endl;
    }
    if (!useBackUpStatus)
        return cam_->InverseProject(GetHostFrameObv(), GetPositiveDepth(invZ_));

    return cam_->InverseProject(GetHostFrameObv(), GetPositiveDepth(invZback_));
}

Eigen::Vector3d Landmark::GetPw(const bool useBackUpStatus) const {
    if (useBackUpStatus) {
        return host_->TwcBack_ * GetPc(true);
    } else {
        return host_->Twc_ * GetPc(false);
    }
}

int Landmark::Size() const {
    return 1;
}

void Landmark::Update(const double delta_z) {
    invZ_ += delta_z;
}

void Landmark::SetTriangulateResult(const double invZ) {
#if 0
    // 使用真值深度进行测试
    if (trueDepth_ < 0.1) {
        // 使用真值深度来验证BA优化有效性
        return;
    }
    invZback_ = invZ_ = 1.0 / trueDepth_; // invZ;
#else
    invZback_ = invZ_ = invZ;
#endif
    obvTime_ = 1;
    failInitializeNum_ = 0;
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
    if (canBedelete_) {
        return false;
    }
    // 同时考虑转移未初始化和已初始化的量
    if (initialized_) {
        const Eigen::Vector3d pc1 = GetPc();
        const Pose T21 = kf2->Tcw_ * host_->Twc_;
        const Eigen::Vector3d pc2 = T21 * pc1;
        if (pc2.z() < kMinSceneDepthInCamera) {
            return false;
        }
        invZback_ = invZ_ = 1.0 / pc2.z();
    }

    // 不在这里删除，因为构建边缘化信息需要
    //if (target_.count(host_)) {
    //    target_.erase(host_);
    //}
    host_ = kf2;
    kpRow_ = target_.at(kf2);
    // TODO：暂不使用首次雅可比
    // kf2->landmark_.emplace_back(this); // kf2已经将所有关键点初始化，这里不能再添加新的landmark
    return true;
}

bool Landmark::TransformHost2NextKeyframe(std::vector<KeyFrame*>& window) {
    // 这里，我们将被边缘化帧的landmark转移到观测到它，且是最新的KF上，
    // 因为对Landmark*进行了传递，所以，直接删除的话，将导致其余KF的core dump
    if (canBedelete_ || target_.size() < 2) {
        return false;
    }

    // window[0]是待移除的kf，win[-1]是最新帧，这里目的是转给下一帧
    for (int i = 1; i < static_cast<int>(window.size()); ++i) {
        KeyFrame* nextKF = window[i];
        if (target_.count(nextKF)) {
            return TransformHost2OtherKF(nextKF);
        }
    }

    return false;
}

void Landmark::CopyStatus() {
    invZback_ = invZ_;
}

void Landmark::BackUpStatus() {
    invZ_ = invZback_;
}

Eigen::Vector2d Landmark::GetHostFrameObv() const {
    return {host_->kpts_(kpRow_, 0), host_->kpts_(kpRow_, 1)};
}
Eigen::Vector2i Landmark::GetHostFrameObvInt() const {
    return {static_cast<int>(host_->kpts_(kpRow_, 0)),
            static_cast<int>(host_->kpts_(kpRow_, 1))};
}
cv::Point2f Landmark::GetHostFrameObvCV() const {
    return {host_->kpts_(kpRow_, 0), host_->kpts_(kpRow_, 1)};
}

//void Landmark::AddKeyframeTargetObv(KeyFrame* kf, const Eigen::Vector2d& obv) {
//    lock_guard<mutex> lock(mute_);
//    target_.insert({kf, obv});
//}

void Landmark::ResetFEJ() {
    //J_Pc2_Twc2.clear();
    //J_Pc2_Pw.clear();
    //J_Pw_z.clear();
    //J_Pw_Twc1.clear();
    //J_Pc2_T12_.clear();
    noUsed_ = false;
    matchNextPixel_.setZero();
}

bool Landmark::Converge() const {
    //if (invZ_ < 1e-9) {
    //    return false;
    //}
    //const double stddev = sqrt(invDepthCov_);
    //return (stddev < 0.001 || stddev / invZ_ < 0.1) && obvTime_ > 5;
    return obvTime_ > config->maxKFnumInWindow;
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
