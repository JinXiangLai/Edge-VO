#include <ceres/ceres.h>  // 包含了ceres全部的头文件

#include <Eigen/Dense>

#include "Pose.h"

// 自定义SE3 LocalParameterization
class SE3Parameterization : public ceres::Manifold {
   public:
    bool Plus(const double* x, const double* delta,
              double* x_plus_delta) const override {
        // 将 delta 转换为四元数
        Eigen::Matrix<double, 3, 1> delta_q(delta[0], delta[1], delta[2]);
        double theta = delta_q.norm();
        Eigen::Quaterniond deltaQ;
        if (theta > 0.0) {
            deltaQ =
                Eigen::Quaterniond(Eigen::AngleAxisd(theta, delta_q / theta));
        } else {
            deltaQ = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
        }

        // 将 x 转换为四元数: qw, qx, qy, qz
        Eigen::Quaterniond x_q(x[0], x[1], x[2], x[3]);

        // 更新四元数
        Eigen::Quaterniond x_plus_delta_q = x_q * deltaQ;
        x_plus_delta_q.normalize();

        // 将结果写回
        x_plus_delta[0] = x_plus_delta_q.w();
        x_plus_delta[1] = x_plus_delta_q.x();
        x_plus_delta[2] = x_plus_delta_q.y();
        x_plus_delta[3] = x_plus_delta_q.z();

        // 更新3D位置
        x_plus_delta[4] = x[4] + delta[3];
        x_plus_delta[5] = x[5] + delta[4];
        x_plus_delta[6] = x[6] + delta[5];

        return true;
    }

    virtual bool ComputeJacobian(const double* x, double* jacobian) const {
        return false;
    }

    virtual bool RightMultiplyByPlusJacobian(
        const double* x, const int num_rows, const double* ambient_matrix,
        double* tangent_matrix) const override {
        return false;
    }

    virtual bool Minus(const double* x, const double* y,
                       double* y_minus_x) const override {
        return false;
    }

    virtual bool PlusJacobian(const double* x,
                              double* jacobian) const override {
        // 计算的是Plus函数对delta增量的雅可比矩阵

        // 因为我在Evaluate函数中已经直接给出∂e/∂δ，所以这里只需要保证乘上∂e/∂δ值不变即可
        // 但此时只能使用ceres::LINE_SEARCH，而不能使用ceres::TRUST_REGION
        // 这里的雅可比行数是7,但是我的目的只需要前6行是单位矩阵即可
        Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
        j.setZero();
        j.block(0, 0, 6, 6).setIdentity();

        return true;
    }

    virtual bool MinusJacobian(const double* x,
                               double* jacobian) const override {
        return false;
    }

    int AmbientSize() const override { return 7; }  // Pose的环境维度
    int TangentSize() const override { return 6; }  // Pose的切空间维度
};

// 自定义SE3 LocalParameterization
class Sim3Parameterization : public ceres::Manifold {
   public:
    bool Plus(const double* x, const double* delta,
              double* x_plus_delta) const override {
        // 将 delta 转换为四元数
        Eigen::Matrix<double, 3, 1> delta_q(delta[0], delta[1], delta[2]);
        double theta = delta_q.norm();
        Eigen::Quaterniond deltaQ;
        if (theta > 0.0) {
            deltaQ =
                Eigen::Quaterniond(Eigen::AngleAxisd(theta, delta_q / theta));
        } else {
            deltaQ = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
        }

        // 将 x 转换为四元数: qw, qx, qy, qz
        Eigen::Quaterniond x_q(x[0], x[1], x[2], x[3]);

        // 更新四元数
        Eigen::Quaterniond x_plus_delta_q = x_q * deltaQ;
        x_plus_delta_q.normalize();

        // 将结果写回
        x_plus_delta[0] = x_plus_delta_q.w();
        x_plus_delta[1] = x_plus_delta_q.x();
        x_plus_delta[2] = x_plus_delta_q.y();
        x_plus_delta[3] = x_plus_delta_q.z();

        // 更新3D位置
        x_plus_delta[4] = x[4] + delta[3];
        x_plus_delta[5] = x[5] + delta[4];
        x_plus_delta[6] = x[6] + delta[5];

        // 更新scale
        x_plus_delta[7] = x[7] + delta[6];

        return true;
    }

    virtual bool ComputeJacobian(const double* x, double* jacobian) const {
        return false;
    }

    virtual bool RightMultiplyByPlusJacobian(
        const double* x, const int num_rows, const double* ambient_matrix,
        double* tangent_matrix) const override {
        return false;
    }

    virtual bool Minus(const double* x, const double* y,
                       double* y_minus_x) const override {
        return false;
    }

    virtual bool PlusJacobian(const double* x,
                              double* jacobian) const override {
        // 计算的是Plus函数对delta增量的雅可比矩阵

        // 因为我在Evaluate函数中已经直接给出∂e/∂δ，所以这里只需要保证乘上∂e/∂δ值不变即可
        // 但此时只能使用ceres::LINE_SEARCH，而不能使用ceres::TRUST_REGION
        // 这里的雅可比行数是7,但是我的目的只需要前6行是单位矩阵即可
        Eigen::Map<Eigen::Matrix<double, 8, 7, Eigen::RowMajor>> j(jacobian);
        j.setZero();
        j.block(0, 0, 7, 7).setIdentity();

        return true;
    }

    virtual bool MinusJacobian(const double* x,
                               double* jacobian) const override {
        return false;
    }

    int AmbientSize() const override { return 8; }  // Pose的环境维度
    int TangentSize() const override { return 7; }  // Pose的切空间维度
};

// 分别代表残差维度，第一个参数块维度Twc1，第2个参数块维度Twc2以及逆深度
// 旋转使用四元数参数化
class ProjectInvDepthResidual : public ceres::SizedCostFunction<2, 7, 7, 1> {
   public:
    ProjectInvDepthResidual(const Eigen::Vector2d& obvHost,
                            const Eigen::Vector2d& obvFrame,
                            const Eigen::Matrix3d& K);
    ProjectInvDepthResidual(
        const Eigen::Vector2d& obvHost,
        const Eigen::Matrix<double, 1, 2, Eigen::RowMajor>& obvFrame,
        const Eigen::Matrix3d& K);
    virtual bool Evaluate(double const* const* parameters, double* residuals,
                          double** jacobians) const override;

   private:
    Eigen::Vector2d obvHost_;
    Eigen::Vector2d obvFrame_;
    Eigen::Matrix3d K_;
};

class ProjectionResidual : public ceres::SizedCostFunction<2, 7> {
   public:
    ProjectionResidual(const Eigen::Vector3d& pw, const Eigen::Vector2d& obv,
                       const Eigen::Matrix3d& K);
    virtual bool Evaluate(double const* const* parameters, double* residuals,
                          double** jacobians) const override;

   private:
    Eigen::Vector3d pw_;
    Eigen::Vector2d obv_;
    Eigen::Matrix3d K_;
};

// 8表示实际需要的参数个数，主要是四元数需要4个表示
class RelativeConstraintResidual : public ceres::SizedCostFunction<7, 8, 8> {
   public:
    RelativeConstraintResidual(const double rotWeight, const double transWeight,
                               const double scaleWeight,
                               const Sim3Pose& sPriorT12);
    virtual bool Evaluate(double const* const* parameters, double* residuals,
                          double** jacobians) const override;

   private:
    double rotWeight_ = 1.0;
    double transWeight_ = 0.1;
    double scaleWeight_ = 1.0;
    Sim3Pose sPriorT12_;
};

class Sim3TransformResidual : public ceres::SizedCostFunction<3, 8> {
   public:
    Sim3TransformResidual(const Eigen::Vector3d& pc1,
                          const Eigen::Vector3d& pc2);
    virtual bool Evaluate(double const* const* parameters, double* residuals,
                          double** jacobians) const override;

   private:
    Eigen::Vector3d pc1_;
    Eigen::Vector3d pc2_;
};
