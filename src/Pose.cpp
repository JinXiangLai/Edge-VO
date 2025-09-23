#include "Pose.h"
#include "Utils.h"

using namespace std;

Pose::Pose(const Pose& T) {
    q_wb_ = T.q_wb_;
    t_wb_ = T.t_wb_;
}

Pose::Pose(const Eigen::Quaterniond& q_wb, const Eigen::Vector3d& t_wb)
    : q_wb_(q_wb), t_wb_(t_wb) {}

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
    q_wb_ *= Exp<double>(delta_q);  // Sophus库标准更新法
    t_wb_ += delta_t;
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
