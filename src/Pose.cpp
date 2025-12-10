#include "Pose.h"
#include "Utils.h"

using namespace std;

Pose::Pose(const Pose& T) {
    q_wb_ = T.q_wb_;
    t_wb_ = T.t_wb_;
}

Pose::Pose(const Eigen::Quaterniond& q_wb, const Eigen::Vector3d& t_wb)
    : q_wb_(q_wb), t_wb_(t_wb) {
    q_wb_.normalize();
}

Pose Pose::Inverse() const {
    Eigen::Quaterniond q_bw = q_wb_.inverse();
    q_bw.normalize();
    const Eigen::Vector3d t_bw = -(q_bw * t_wb_);
    return Pose(q_bw, t_bw);
}

Pose Pose::operator*(const Pose& T) const {
    return Pose(q_wb_ * T.q_wb_, q_wb_ * T.t_wb_ + t_wb_);
}

Eigen::Vector3d Pose::operator*(const Eigen::Vector3d& p) const {
    return q_wb_ * p + t_wb_;
}

int Pose::Size() const {
    return 6;
}

void Pose::Update(const Eigen::Vector3d& delta_q,
                  const Eigen::Vector3d& delta_t) {
    // const Eigen::Matrix3d deltaR = Eigen::AngleAxisd(delta_q.norm(), delta_q.normalized()).toRotationMatrix();
    // q_wb_ *= Eigen::Quaterniond(deltaR);
    // t_wb_ += q_wb_ * delta_t;              // 不能使用优化前的q更新
    q_wb_ = q_wb_ * Exp<double>(delta_q);  // Sophus库标准更新法
    q_wb_.normalize();
    t_wb_ += delta_t;
    // t_wb_ += q_wb_ * delta_t;  // 也不能这样使用优化后的q更新
    // 因为我是分开求导的，而不是使用SE3群求导
}

Eigen::Matrix4d Pose::ToMatrix4d() const {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block(0, 0, 3, 3) = q_wb_.toRotationMatrix();
    T.block(0, 3, 3, 1) = t_wb_;
    return T;
}

std::string Pose::QwbString() const {
    return fmt::format("Qwb: {:.2f}, {:.2f}, {:.2f}, {:.2f}", q_wb_.w(),
                       q_wb_.x(), q_wb_.y(), q_wb_.z());
}

std::string Pose::PwbString() const {
    return fmt::format("Pwb: {:.2f}, {:.2f}, {:.2f}", t_wb_.x(), t_wb_.y(),
                       t_wb_.z());
}

// ========== Sim3Pose ================//
Sim3Pose::Sim3Pose(const Sim3Pose& T) {
    q_wb_ = T.q_wb_;
    t_wb_ = T.t_wb_;
    scale_ = T.scale_;
}

Sim3Pose::Sim3Pose(const Pose& T, const double scale) {
    q_wb_ = T.q_wb_;
    t_wb_ = T.t_wb_;
    scale_ = scale;
}

Sim3Pose::Sim3Pose(const Eigen::Quaterniond& q_wb, const Eigen::Vector3d& t_wb,
                   const double scale)
    : q_wb_(q_wb), t_wb_(t_wb), scale_(scale) {}

Sim3Pose Sim3Pose::Inverse() const {
    Eigen::Quaterniond q_bw = q_wb_.inverse();
    q_bw.normalize();
    const double invS = 1.0 / scale_;
    const Eigen::Vector3d t_bw = -invS * (q_bw * t_wb_);
    return Sim3Pose(q_bw, t_bw, invS);
}

Sim3Pose Sim3Pose::operator*(const Sim3Pose& T) const {
    return Sim3Pose(q_wb_ * T.q_wb_, scale_ * (q_wb_ * T.t_wb_) + t_wb_,
                    scale_ * T.scale_);
}

Eigen::Vector3d Sim3Pose::operator*(const Eigen::Vector3d& p) const {
    return scale_ * (q_wb_ * p) + t_wb_;
}

int Sim3Pose::Size() const {
    return 7;
}

void Sim3Pose::Update(const Eigen::Vector3d& delta_q,
                      const Eigen::Vector3d& delta_t, const double delta_s) {
    // | s_old * R_old, t_old |   | Δs * ΔR, Δt |
    // | 0,             1     | * | 0,       1  |
    // =>
    // R_new = R_old * ΔR = R_old * Exp(Δξ)
    // t_new = t_old
    // 但是这里是分开求导，因此不能用Sim3群更新规则
    q_wb_ = q_wb_ * Exp<double>(delta_q);  // Sophus库标准更新法
    q_wb_.normalize();
    t_wb_ += delta_t;
    scale_ += delta_s;
}

Eigen::Matrix4d Sim3Pose::ToMatrix4d() const {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block(0, 0, 3, 3) = scale_ * q_wb_.toRotationMatrix();
    T.block(0, 3, 3, 1) = t_wb_;
    return T;
}

std::string Sim3Pose::QwbString() const {
    return fmt::format("Qwb: {:.2f}, {:.2f}, {:.2f}, {:.2f}", q_wb_.w(),
                       q_wb_.x(), q_wb_.y(), q_wb_.z());
}

std::string Sim3Pose::PwbString() const {
    return fmt::format("Pwb: {:.2f}, {:.2f}, {:.2f}, scale: {:.2f}", t_wb_.x(),
                       t_wb_.y(), t_wb_.z(), scale_);
}

void Sim3Pose::CopyStatus() {
    q_wb_back_ = q_wb_;
    t_wb_back_ = t_wb_;
    scale_back_ = scale_;
}

void Sim3Pose::BackUpStatus() {
    q_wb_ = q_wb_back_;
    t_wb_ = t_wb_back_;
    scale_ = scale_back_;
}

string Sim3Pose::DebugOutputPoseMessage() const {
    const Eigen::Vector3d p = t_wb_ / scale_;
    const Eigen::Quaterniond& q = q_wb_;
    return fmt::format("{:.6f} {} {} {} {} {} {} {}", debugTimestamp_, p.x(),
                       p.y(), p.z(), q.x(), q.y(), q.z(), q.w());
}