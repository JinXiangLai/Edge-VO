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

RelativeConstraintResidual::RelativeConstraintResidual(
    const double rotWeight, const double transWeight, const double scaleWeight,
    const Sim3Pose& sPriorT12)
    : rotWeight_(rotWeight),
      transWeight_(transWeight),
      scaleWeight_(scaleWeight),
      sPriorT12_(sPriorT12) {}

bool RelativeConstraintResidual::Evaluate(double const* const* parameters,
                                          double* residuals,
                                          double** jacobians) const {
    const double* q1 = parameters[0];
    const double* p1 = parameters[0] + 4;
    const double s1 = *(parameters[0] + 7);
    const Eigen::Quaterniond Qwc1(q1[0], q1[1], q1[2], q1[3]);
    const Eigen::Matrix3d Rwc1 = Qwc1.toRotationMatrix();
    const Eigen::Vector3d Pwc1(p1[0], p1[1], p1[2]);
    const double invS1 = 1.0 / s1;

    const double* q2 = parameters[1];
    const double* p2 = parameters[1] + 4;
    const double s2 = *(parameters[1] + 7);
    const Eigen::Quaterniond Qwc2(q2[0], q2[1], q2[2], q2[3]);
    const Eigen::Matrix3d Rwc2 = Qwc2.toRotationMatrix();
    const Eigen::Vector3d Pwc2(p2[0], p2[1], p2[2]);

    // |s1R1, t1|   |s2R2, t2|
    // |  0,  1 | * |   0,  1| =
    //
    // |s1*s2*R1*R2, s1R1*t2+t1|
    // |          0, 1         |
    // ΔR = LogSO3(R1 * R2)
    // Δt = s1*R1*t2 + t1
    // 需保证和雅可比计算的顺序一致
    // 计算残差向量
    // 注意取逆
    const Eigen::Matrix3d dR = sPriorT12_.q_wb_.toRotationMatrix().transpose();
    const Eigen::Vector3d& dP = sPriorT12_.t_wb_;
    const double ds = sPriorT12_.scale_;
    const Eigen::Vector3d dr = LogSO3(dR * (Rwc1.transpose() * Rwc2));
    residuals[0] = dr[0] * rotWeight_;
    residuals[1] = dr[1] * rotWeight_;
    residuals[2] = dr[2] * rotWeight_;

    // |1/s1*R1.inv, -1/s1*R1.inv*t1|   |s2R2, t2|
    // |          0,               1| * |   0,  1|
    const Eigen::Vector3d dPw = Pwc2 - Pwc1;
    const Eigen::Vector3d sP12 = invS1 * (Qwc1.inverse() * dPw);
    const Eigen::Vector3d dp = sP12 - dP;
    residuals[3] = dp[0] * transWeight_;
    residuals[4] = dp[1] * transWeight_;
    residuals[5] = dp[2] * transWeight_;

    // 注意：这里实现存在的问题是量纲不统一，且scale一定是非负数
    residuals[6] =
        (invS1 * s2 - ds) * scaleWeight_;  // 相邻帧间尺度漂移比例应该接近于1.0

    // ΔR = LogSO3(dR * Rw1.inv * Rw2)
    // ΔT = Twc1.inv * Twc2
    // ΔP = 1/s1 * Rw1.inv * Pw2 - 1/s1 * Rw1.inv * Pw1 - dP
    //    = 1/s1 * Rw1.inv * (Pw2 - Pw1) - dP
    // Δs = 1.0/s1*s2 - ds
    if (jacobians) {
        // 使用"BCH近似"之前，需要通过"伴随性质"将扰动量换到右边
        // dLogSO3(dR * R1.T*R2) ---> 微分扰动
        // = LogSO3(dR * exp(-ε1^)*R1.T*R2) ---> 使用伴随: Exp(ε1)*R = R * Exp(R.T * ε1)
        // = LogSO3(dR*R1.T*R2 * Exp(R2.T*R1*-ε1)) ---> Exp{小量}，使用BCH近似
        // = [Jr(dR*R1.T*R2).inv * -R2.T*R1*ε1] + LogSo3(dR*R1.T*R2)
        // 关于Rw2，易得最终的BCH近似为：
        // [Jr(dR*R1.T*R2).inv * ε2] + LogSo3(dR*R1.T*R2)
        const Eigen::Matrix3d invJr = InverseRightJacobianSO3(dr);
        if (jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 7, 8, Eigen::RowMajor>> A1(
                jacobians[0]);
            A1.setZero();
            // ΔR w.r.t Pw1, s1 = 0
            // ΔR w.r.t Rw1
            A1.block<3, 3>(0, 0) =
                -invJr * Rwc2.transpose() * Rwc1 * rotWeight_;

            // ΔP w.r.t Rw1
            A1.block<3, 3>(3, 0) =
                invS1 * SkewSymmetric(Rwc1.transpose() * dPw) * transWeight_;
            // ΔP w.r.t Pw1
            A1.block<3, 3>(3, 3) = -invS1 * Rwc1.transpose() * transWeight_;
            // ΔP w.r.t s1
            A1.block<3, 1>(3, 6) =
                -invS1 * invS1 * Rwc1.transpose() * dPw * transWeight_;

            // Δs w.r.t Rw1, Rw2, Pw1, Pw2 = 0
            // Δs w.r.t s1
            A1(6, 6) = -s2 * invS1 * invS1 * scaleWeight_;
        }
        if (jacobians[1]) {
            Eigen::Map<Eigen::Matrix<double, 7, 8, Eigen::RowMajor>> A2(
                jacobians[1]);
            A2.setZero();
            // ΔR w.r.t Pw2, s2 = 0
            // ΔR w.r.t Rw2
            A2.block<3, 3>(0, 0) = invJr * dR.transpose() * rotWeight_;

            // ΔP w.r.t Rw2 = 0
            // ΔP w.r.t s2 = 0
            // ΔP w.r.t Pw2
            A2.block<3, 3>(3, 3) = invS1 * Rwc1.transpose() * transWeight_;

            // Δs w.r.t s2
            A2(6, 6) = s1 * scaleWeight_;
        }
    }

    return true;
}

Sim3TransformResidual::Sim3TransformResidual(const Eigen::Vector3d& pc1,
                                             const Eigen::Vector3d& pc2)
    : pc1_(pc1), pc2_(pc2) {}

bool Sim3TransformResidual::Evaluate(double const* const* parameters,
                                     double* residuals,
                                     double** jacobians) const {
    // sT12
    const double* q = parameters[0];
    const double* p = parameters[0] + 4;
    const double s = *(parameters[0] + 7);

    const Eigen::Quaterniond Q12(q[0], q[1], q[2], q[3]);
    const Eigen::Matrix3d R12 = Q12.toRotationMatrix();
    const Eigen::Vector3d P12(p[0], p[1], p[2]);

    const Eigen::Vector3d rotPc2 = Q12 * pc2_;
    const Eigen::Vector3d pc1 = s * rotPc2 + P12;
    const Eigen::Vector3d r = pc1 - pc1_;

    residuals[0] = r[0];
    residuals[1] = r[1];
    residuals[2] = r[2];

    if (jacobians) {
        if (jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 3, 8, Eigen::RowMajor>> J_res_sT12(
                jacobians[0]);
            J_res_sT12.setZero();
            // res w.r.t R12
            J_res_sT12.block<3, 3>(0, 0) = -s * R12 * SkewSymmetric(pc2_);
            // res w.r.t P12
            J_res_sT12.block<3, 3>(0, 3).setIdentity();
            // res w.r.t s12
            J_res_sT12.block<3, 1>(0, 6) = rotPc2;
        }
    }

    return true;
}
