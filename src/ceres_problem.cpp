#include "ceres_problem.h"

#include "Utils.h"
using namespace std;

ProjectInvDepthResidual::ProjectInvDepthResidual(
    const Eigen::Vector2d& obvHost, const Eigen::Vector2d& obvFrame,
    const Eigen::Matrix3d& K)
    : obvHost_(obvHost), obvFrame_(obvFrame), K_(K) {
    // 已在继承类指定const，这里就不需要再指定
}

ProjectInvDepthResidual::ProjectInvDepthResidual(
    const Eigen::Vector2d& obvHost,
    const Eigen::Matrix<double, 1, 2, Eigen::RowMajor>& obvFrame,
    const Eigen::Matrix3d& K)
    : obvHost_(obvHost), K_(K) {
    obvFrame_[0] = obvFrame[0];
    obvFrame_[1] = obvFrame[1];
}

bool ProjectInvDepthResidual::Evaluate(double const* const* parameters,
                                       double* residuals,
                                       double** jacobians) const {
    // 获取优化参数
    const double* q_wc1 = parameters[0];  // qw, qx, qy, qz
    const double* p_wc1 = parameters[0] + 4;
    const double* q_wc2 = parameters[1];
    const double* p_wc2 = parameters[1] + 4;

    const Eigen::Quaterniond Qwc1(q_wc1[0], q_wc1[1], q_wc1[2], q_wc1[3]);
    const Eigen::Vector3d Pwc1(p_wc1[0], p_wc1[1], p_wc1[2]);
    const Eigen::Quaterniond Qwc2(q_wc2[0], q_wc2[1], q_wc2[2], q_wc2[3]);
    const Eigen::Vector3d Pwc2(p_wc2[0], p_wc2[1], p_wc2[2]);
    const double& invZ1 = *parameters[2];
    const double z1 = 1.0 / invZ1;

    // 重投影
    const Eigen::Vector3d pc1Norm((obvHost_.x() - K_(0, 2)) / K_(0, 0),
                                  (obvHost_.y() - K_(1, 2)) / K_(1, 1), 1.0);
    const Eigen::Vector3d pc1 = z1 * pc1Norm;
    const Eigen::Vector3d pw = Qwc1 * pc1 + Pwc1;
    const Eigen::Vector3d dPw = pw - Pwc2;
    const Eigen::Quaterniond Qc2w = Qwc2.inverse();
    const Eigen::Vector3d pc2 = Qc2w * dPw;
    const Eigen::Vector3d pc2Norm = pc2 / pc2.z();
    const Eigen::Vector2d projectObv2(pc2Norm.x() * K_(0, 0) + K_(0, 2),
                                      pc2Norm.y() * K_(1, 1) + K_(1, 2));

    // 记录残差
    residuals[0] = projectObv2[0] - obvFrame_[0];
    residuals[1] = projectObv2[1] - obvFrame_[1];

    // 计算雅可比
    if (jacobians) {
        // clang-format off
            /******** 投影过程 ********
            * K.inv * (u1, v1, 1) --> Pc1_norm * z1 --> Twc1 * Pc1 --> Twc2.inv * Pw -->  
            *  Pc2 / z2 -> K * Pc2_norm -> (u2, v2, 1) -> res(u2, v2)
            *
            * res w.r.t (u2, v2) [1x2]
            * (u2, v2) w.r.t Pc2_norm [2x3]
            * Pc2_norm w.r.t Pc2 [3x3]
            * Pc2 w.r.t Twc2 [3x6] ------> optimization variable
            * Pc2 w.r.t Pw [3x3]
            * Pw w.r.t Twc1 [3x6] -------> optimization variable
            * Pw w.r.t Pc1
            * Pc1 w.r.t z1 [3x1] --------> optimization variable
            ************************/
        // clang-format on
        const auto& J_px2_Pc2Norm =
            K_.block(0, 0, 2, 3);  // 第3列可不置0,因为后续J_Pc2Norm_Pc2第3行为0

        Eigen::Matrix<double, 3, 3> J_Pc2Norm_Pc2;
        const double d = 1 / pc2.z();
        const double d2 = d * d;
        J_Pc2Norm_Pc2 << d, 0, -pc2.x() * d2, 0, d, -pc2.y() * d2, 0, 0, 0;

        const Eigen::Matrix<double, 2, 3> J_px2_Pc2 =
            J_px2_Pc2Norm * J_Pc2Norm_Pc2;
        const Eigen::Matrix<double, 2, 3>& J_res_Pc2 = J_px2_Pc2;

        Eigen::Matrix<double, 3, 6> J_Pc2_Twc2;
        // Pc2 w.r.t Rwc2
        J_Pc2_Twc2.block<3, 3>(0, 0) = SkewSymmetric(Qc2w * dPw);
        // Pc2 w.r.t Pwc2
        const Eigen::Matrix3d Rc2w = Qc2w.toRotationMatrix();
        J_Pc2_Twc2.block<3, 3>(0, 3) = -Rc2w;

        // Pc2 w.r.t Pw
        const Eigen::Matrix3d& J_Pc2_Pw = Rc2w;

        Eigen::Matrix<double, 3, 6> J_Pw_Twc1;
        const Eigen::Matrix3d Rwc1 = Qwc1.toRotationMatrix();
        // Pw w.r.t Twc1 : Pw = Twc1 * Pc1 = Rwc1 * pc1 + Pwc1
        // Pw w.r.t Rwc1
        J_Pw_Twc1.block<3, 3>(0, 0).noalias() = -Rwc1 * SkewSymmetric(pc1);

        // Pw w.r.t Pwc1
        J_Pw_Twc1.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();

        // Pw w.r.t Pc1
        const Eigen::Matrix3d& J_Pw_Pc1 = Rwc1;

        // Pc1 w.r.t z
        // 使用逆深度表示
        const double invZ1Square = invZ1 * invZ1;
        Eigen::Vector3d J_Pc1_z;
        J_Pc1_z << -pc1Norm.x() / invZ1Square, -pc1Norm.y() / invZ1Square,
            -1 / invZ1Square;

        const Eigen::Matrix<double, 3, 1> J_Pw_z = J_Pw_Pc1 * J_Pc1_z;

        // Residual w.r.t optimization variables Jacobian
        if (jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J_res_Twc1(
                jacobians[0]);
            // 由于维度是关于四元数的，所以需要设置zero
            J_res_Twc1.setZero();
            J_res_Twc1.block<2, 6>(0, 0) = J_res_Pc2 * J_Pc2_Pw * J_Pw_Twc1;
        }

        if (jacobians[1]) {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J_res_Twc2(
                jacobians[1]);
            J_res_Twc2.setZero();
            J_res_Twc2.block<2, 6>(0, 0) = J_res_Pc2 * J_Pc2_Twc2;
        }

        if (jacobians[2]) {
            Eigen::Map<Eigen::Matrix<double, 2, 1>> J_res_invZ1(jacobians[2]);
            J_res_invZ1.setZero();
            J_res_invZ1 = J_res_Pc2 * J_Pc2_Pw * J_Pw_z;
        }
    }

    return true;
}

ProjectionResidual::ProjectionResidual(const Eigen::Vector3d& p_w,
                                       const Eigen::Vector2d& obv,
                                       const Eigen::Matrix3d& K)
    : pw_(p_w), obv_(obv), K_(K) {}

bool ProjectionResidual::Evaluate(double const* const* parameters,
                                  double* residuals, double** jacobians) const {
    // 获取优化参数
    const double* q_wc2 = parameters[0];  // qw, qx, qy, qz
    const double* p_wc2 = parameters[0] + 4;
    const Eigen::Quaterniond Qwc2(q_wc2[0], q_wc2[1], q_wc2[2], q_wc2[3]);
    const Eigen::Vector3d Pwc2(p_wc2[0], p_wc2[1], p_wc2[2]);

    const Eigen::Vector3d dPw = pw_ - Pwc2;
    const Eigen::Vector3d pc2 = Qwc2.inverse() * dPw;
    const Eigen::Vector3d pc2Norm = pc2 / pc2.z();
    const Eigen::Vector2d projectObv2(pc2Norm.x() * K_(0, 0) + K_(0, 2),
                                      pc2Norm.y() * K_(1, 1) + K_(1, 2));

    // 记录残差
    residuals[0] = projectObv2[0] - obv_[0];
    residuals[1] = projectObv2[1] - obv_[1];

    // 计算雅可比
    if (jacobians && jacobians[0]) {
        const auto& J_px2_Pc2Norm = K_.block(0, 0, 2, 3);
        Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J_res_Twc2(
            jacobians[0]);
        // 由于维度是关于四元数的，所以需要设置zero
        J_res_Twc2.setZero();
        const double d = 1 / pc2.z();
        const double d2 = d * d;
        Eigen::Matrix<double, 3, 3> J_Pc2Norm_Pc2;
        J_Pc2Norm_Pc2 << d, 0, -pc2.x() * d2, 0, d, -pc2.y() * d2, 0, 0, 0;

        const Eigen::Matrix<double, 2, 3> J_px2_Pc2 =
            J_px2_Pc2Norm * J_Pc2Norm_Pc2;

        const Eigen::Quaterniond Qc2w = Qwc2.inverse();
        Eigen::Matrix<double, 3, 6> J_Pc2_Twc2;
        // * Pc2 w.r.t t12
        J_Pc2_Twc2.block<3, 3>(0, 3) = -Qc2w.toRotationMatrix();
        J_Pc2_Twc2.block<3, 3>(0, 0) = SkewSymmetric(Qc2w * dPw);

        J_res_Twc2.block<2, 6>(0, 0) = J_px2_Pc2 * J_Pc2_Twc2;
    }

    return true;
}
