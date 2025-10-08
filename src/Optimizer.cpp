#include "Optimizer.h"

#include <stdlib.h>
#include <unistd.h>
#include <cfloat>
#include <random>
#include <set>

#include "Config.h"
#include "Eigen/src/Core/Matrix.h"
#include "KeyFrame.h"
#include "Landmark.h"
#include "Utils.h"

#define USE_DT_RESIDUAL  // 测试优化算法是否有问题

using namespace std;
using namespace cv;

constexpr double kPriorDepthWeight = 1e8;

Optimizer::Optimizer(shared_ptr<Camera> cam, const double lambda,
                     const int maxIte, const bool onlyPoseUpdate)
    : lambda_(lambda),
      maxIte_(maxIte),
      onlyPoseUpdate_(onlyPoseUpdate),
      cam_(cam) {
    maxIte_ = config->maxIteration;
}

Optimizer::~Optimizer() {
    // 排查内存泄漏
    for (KeyFrame* kf : window_) {
        if (kf != nullptr) {
            delete kf;
            kf = nullptr;
        }
    }
}

Optimizer::ResidualInfo Optimizer::CalculateResidualCurFrame(
    const std::vector<Landmark*>& lk1s,
    const std::vector<Eigen::Vector2d>& obvs, const Pose& Twc2,
    const cv::Mat& img, const bool checkAbnormalLandmark) {
    chrono::steady_clock::time_point t0 = chrono::steady_clock::now();

    ResidualInfo info;
    string debugInfo("chi2 residuals: ");
    int stepInfoOut = 30;

    const Camera& cam = *cam_;
    const Pose Tc2w = Twc2.Inverse();
    const double maxChi2 = config->maxProjectError * config->maxProjectError;
    for (size_t j = 0; j < lk1s.size(); ++j) {
        Landmark* lk1 = lk1s[j];
        if (lk1->noUsed_ || !lk1->initialized_) {
            continue;
        }
        const Eigen::Vector3d pc = Tc2w * lk1->GetPw();
        const Eigen::Vector2d px = cam.Project2PixelPlane(pc);
        const bool inRange = InRange(img, px.cast<int>());
        if (!checkAbnormalLandmark ||
            (inRange && pc.z() > kMinSceneDepthInCamera)) {
            // 必须与计算Jacobian的残差计算方式一致
            // 注意： cost = p.T * p = [1x1]向量，我们是对cost进行线性化，因此求导的对象是r^2，
            // 而胡伯核函数的自变量是r^2
            double chi2 = (px - obvs[j]).squaredNorm();
            if (chi2 > maxChi2) {
                lk1->noUsed_ = true;
                continue;
            }
            Eigen::Vector2d rho;  // 残差值和核函数关于残差的导数
            HuberLoss(chi2, rho);
            info.cost += rho[0];
            ++info.usefulNum;
            if (j % stepInfoOut == 0) {
                debugInfo.append(fmt::format(
                    "chi2: {:.1f}-rho[0]: {:.1f}-kf id: {}-kp1:({:.0f}, "
                    "{:.0f}); ",
                    chi2, rho[0], lk1->host_->id_, lk1->uv_.x(), lk1->uv_.y()));
            }
        } else if (checkAbnormalLandmark) {
            // 优化后，使得有些点投影到了图像外，这个时候，我们需要将其设置为 noUsed，
            // 并且，需要使用优化前的pose重新计算残差，
            // 优化后的pose产生的离群点总是存在滞后

            // 这一步，理论上只会被优化后的{poses, landmarks}执行
            lk1->noUsed_ = true;
            continue;
        }
    }

    cout << fixed << "residual info.all.cost: " << info.cost
         << " useful num: " << info.usefulNum << "\n";
    cout << fmt::format("debug residual info: {}\n", debugInfo);
    info.cost /= info.usefulNum;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

    cout << "noInrangeNum | mean cost | spend time: "
         << (lk1s.size() - info.usefulNum) << " | " << info.cost << " | "
         << chrono::duration<double>(t1 - t0).count() << "s" << endl;
    if (isnan(info.cost) || isinf(info.cost)) {
        info.cost = DBL_MAX;
    }
    return info;
}

Optimizer::ResidualInfo Optimizer::CalculateJacobianAndCost(
    vector<Landmark*>& lk1s, const vector<Pose>& T12, Eigen::MatrixXd& H,
    Eigen::VectorXd& g, const int lvl) {

    constexpr int resDim = 1;
    int optVariableDim = -1;
    if (!onlyPoseUpdate_) {
        optVariableDim =
            T12.size() * T12[0].Size() + lk1s.size() * lk1s[0]->Size();
    } else {
        optVariableDim = T12.size() * T12[0].Size();
    }

    ResidualInfo info;
    H.resize(optVariableDim, optVariableDim);
    H.setZero();
    g.resize(optVariableDim);
    g.setZero();

    /******** 投影过程 ********
    * K.inv * (u1, v1, 1) -> Pc1_norm * z1 -> T12.inv * Pc1 -> Pc2 / z2 -> K * Pc2_norm -> (u2, v2, 1) -> res(u2, v2)
    * res w.r.t (u2, v2) [1x2]
    * (u2, v2) w.r.t Pc2_norm [2x3]
    * Pc2_norm w.r.t Pc2 [3x3]
    * Pc2 w.r.t T12 [3x6] ------> optimization variable
    * Pc2 w.r.t Pc1 [3x3]
    * Pc1 w.r.t z1 [3x1] -------> optimization variable
    ************************/
    for (int i = 0; i < T12.size(); ++i) {
        const int poseStartCol = T12[0].Size() * i;
        int pointStartCol = T12.size() * T12[0].Size();
        const Pose T21 = T12[i].Inverse();

        for (int j = 0; j < lk1s.size(); ++j) {
            Landmark& p = *lk1s[j];
            if (p.noUsed_) {
                continue;
            }
            const Eigen::Vector3d Pc1 = p.GetPc();
            const Eigen::Vector3d Pc2 = T21 * Pc1;
            const Eigen::Vector2d px2 = p.cam_->Project2PixelPlane(Pc2, lvl);
            if (!InRange(p.host_->grayImg_, px2.cast<int>())) {
                p.noUsed_ = true;
                continue;
            }

            const Eigen::Vector2d r = px2 - p.uv_;
            double chi2 = r.squaredNorm();
            Eigen::Vector2d rho;
            HuberLoss(chi2, rho, lvl);
            info.cost += rho[0];
            ++info.usefulNum;

            // res w.r.t px2 [1x2], [2x2]的单位矩阵

            // px2 w.r.t Pc2 [2x3]
            Eigen::Matrix<double, 2, 3> J_px2_Pc2Norm =
                p.cam_->K_[lvl].block(0, 0, 2, 3);
            Eigen::Matrix<double, 3, 3> J_Pc2Norm_Pc2;
            const double d = 1 / Pc2.z();
            const double d2 = 1. / pow(Pc2.z(), 2);
            J_Pc2Norm_Pc2 << d, 0, -Pc2.x() * d2, 0, d, -Pc2.y() * d2, 0, 0, 0;
            Eigen::Matrix<double, 2, 3> J_px2_Pc2 =
                J_px2_Pc2Norm * J_Pc2Norm_Pc2;

            // Pc2 w.r.t T12 [3x6]
            Eigen::Matrix<double, 3, 6> J_Pc2_T12 =
                Eigen::Matrix<double, 3, 6>::Zero();
            // FEJ
            // TODO: 使用不同的lvl层时，需要重置FEJ
            if (p.J_Pc2_T12_.empty()) {
                const Eigen::Vector3d dt = Pc1 - T12[i].t_wb_;
                // * Pc2 w.r.t R12
                J_Pc2_T12.block(0, 0, 3, 3) = SkewSymmetric(T21.q_wb_ * dt);
                // * Pc2 w.r.t t12
                J_Pc2_T12.block(0, 3, 3, 3) = -T21.q_wb_.toRotationMatrix();
                if (p.J_Pc2_T12_.empty())
                    p.J_Pc2_T12_.push_back(J_Pc2_T12);  // debug
            } else {
                J_Pc2_T12 = p.J_Pc2_T12_[0];
            }

            Eigen::Matrix<double, 3, 3> J_Pc2_Pc1;
            Eigen::Matrix<double, 3, 1> J_Pc1_z1;
            // TODO: 实现FEJ
            if (!onlyPoseUpdate_) {
                // Pc2 w.r.t Pc1 [3x3]
                J_Pc2_Pc1 = T21.q_wb_.toRotationMatrix();

                // Pc1 w.r.t z1 [3x1]
                const Eigen::Vector3d pc1Norm(p.GetPcNorm());
                J_Pc1_z1 << pc1Norm.x(), pc1Norm.y(), 1;
                // 使用逆深度表示
                const double d = pow(p.invZ_, 2);
                J_Pc1_z1 << -pc1Norm.x() / d, -pc1Norm.y() / d, -1 / d;
            }

            // 给整体雅可比矩阵赋值
            // TODO： 不需要Fixed pose，只要相对Pose准确
            // H = J'*J, g = -J'*b;
            const int ai = i * lk1s.size() + j, aj = poseStartCol;
            const int bi = i * lk1s.size() + j, bj = pointStartCol + j;
            // cout << "J_res_px2:\n" << J_res_px2 << endl;
            // cout << "J_px2_Pc2:\n" << J_px2_Pc2 << endl;
            // cout << "J_Pc2_T12:\n" << J_Pc2_T12 << endl;
            Eigen::Matrix<double, 2, 6> A = J_px2_Pc2 * J_Pc2_T12;
            // A.setZero();

            double w = 1.0;  // 1.0 / lk1s[i]->invDepthCov_;

            H.block(aj, aj, A.cols(), A.cols()) +=
                A.transpose() * A * w * rho[1];
            /******** -J.T * b的size为[J.cols() x 1]**************
            * | A.T  C.T  E.T |       | A.T*b1 + C.T*b2 + E.T*b3|
            * | B.T  D.T  F.T | * b = | B.T*b1 + D.T*b2 + F.T*b3|
            *
            *****************************************************/
            g.middleRows(aj, A.cols()) -= A.transpose() * r * w * rho[1];

            if (!onlyPoseUpdate_) {
                Eigen::MatrixXd B = J_px2_Pc2 * J_Pc2_Pc1 * J_Pc1_z1;
                /******** 利用分块及稀疏矩阵性质直接计算H矩阵 ********
                * 否则，H=J.T * J由于没有利用到稀疏性，计算量将异常大
                * | A.T, C.T, E.T |   | A, B|
                * | B.T, D.T, F.T | * | C, D|
                *                     | E, F| = 
                * | A.T*A + C.T*C + E.T*E,  A.T*B + C.T*D + E.T*F |
                * | B.T*A + D.T*C + F.T*E,  B.T*B + D.T*D + F.T*F |
                * 观察D、E矩阵块的变化规律，可以写出如下的等式
                */
                // TODO:添加胡伯核关于chi2的一阶导数
                H.block(aj, bj, A.cols(), B.cols()) +=
                    A.transpose() * B * w * rho[1];
                H.block(bj, aj, B.cols(), A.cols()) +=
                    B.transpose() * A * w * rho[1];
                H.block(bj, bj, B.cols(), B.cols()) +=
                    B.transpose() * B * w * rho[1];
                g.middleRows(bj, B.cols()) -= B.transpose() * r * w * rho[1];
            }
        }
    }

    info.cost /= info.usefulNum;
    return info;
}

Optimizer::ResidualInfo Optimizer::CalculateJacobianAndCostCurFrame(
    const std::vector<Landmark*>& lk1s,
    const std::vector<Eigen::Vector2d>& obvs, const Pose& Twc2,
    Eigen::MatrixXd& H, Eigen::VectorXd& g) {
    constexpr int resDim = 2;
    int optVariableDim = -1;
    if (!onlyPoseUpdate_) {
        optVariableDim = Twc2.Size() + lk1s.size() * lk1s[0]->Size();
    } else {
        optVariableDim = Twc2.Size();
    }

    ResidualInfo info;
    H.resize(optVariableDim, optVariableDim);
    H.setZero();
    g.resize(optVariableDim);
    g.setZero();
    constexpr double noUpdatePoseNum = 1.0;  // 或者是无穷大？

    /******** 投影过程 ********
    * K.inv * (u1, v1, 1) -> Pc1_norm * z1 -> Twc1 * Pw1 -> Twc2.inv * Pw1 -> Pc2 / z2 -> K * Pc2_norm -> (u2, v2, 1) -> res(u2, v2)
    * res w.r.t (u2, v2) [2x2] 单位矩阵
    * (u2, v2) w.r.t Pc2_norm [2x3]
    * Pc2_norm w.r.t Pc2 [3x3]
    * Pc2 w.r.t Twc2 [3x6] ------> optimization variable
    * Pc2 w.r.t Pw1 [3x3]
    * Pw1 w.r.t Pc1 [3x3]
    * Pc1 w.r.t z1 [3x1] -------> optimization variable
    ************************/
    const int poseStartCol = Twc2.Size() * 0;
    int pointStartCol = Twc2.Size();
    const Pose Tc2w = Twc2.Inverse();

    // res w.r.t px2 [2x2]的单位矩阵
    // px2 w.r.t Pc2 [2x3]
    Eigen::Matrix<double, 2, 3> J_px2_Pc2Norm =
        lk1s[0]->cam_->K_[0].block(0, 0, 2, 3);

    for (size_t j = 0; j < lk1s.size(); ++j) {
        Landmark& p = *lk1s[j];
        if (p.noUsed_ || !p.initialized_) {
            continue;
        }
        const Eigen::Vector3d Pw1 = p.GetPw();
        const Eigen::Vector3d Pc2 = Tc2w * Pw1;
        const Eigen::Vector2d px2 = p.cam_->Project2PixelPlane(Pc2);
        //if (!InRange(p.host_->grayImg_, px2.cast<int>()) ||
        //    Pc2.z() < kMinSceneDepthInCamera) {
        //    p.noUsed_ = true;
        //    continue;
        //}

        const Eigen::Vector2d r = px2 - obvs[j];
        double chi2 = r.squaredNorm();
        Eigen::Vector2d rho;
        HuberLoss(chi2, rho);
        info.cost += rho[0];
        ++info.usefulNum;

        Eigen::Matrix<double, 3, 3> J_Pc2Norm_Pc2;
        const double d = 1 / Pc2.z();
        const double d2 = 1. / pow(Pc2.z(), 2);
        J_Pc2Norm_Pc2 << d, 0, -Pc2.x() * d2, 0, d, -Pc2.y() * d2, 0, 0, 0;
        Eigen::Matrix<double, 2, 3> J_px2_Pc2 = J_px2_Pc2Norm * J_Pc2Norm_Pc2;

        // Pc2 w.r.t T12 [3x6]
        Eigen::Matrix<double, 3, 6> J_Pc2_Twc2 =
            Eigen::Matrix<double, 3, 6>::Zero();
        // FEJ
        if (p.J_Pc2_T12_.empty()) {
            const Eigen::Vector3d dt = Pw1 - Twc2.t_wb_;
            // * Pc2 w.r.t R12
            J_Pc2_Twc2.block(0, 0, 3, 3) = SkewSymmetric(Tc2w.q_wb_ * dt);
            // * Pc2 w.r.t t12
            J_Pc2_Twc2.block(0, 3, 3, 3) = -Tc2w.q_wb_.toRotationMatrix();
            // 不使用FEJ！！！
            //if (p.J_Pc2_T12_.empty())
            //    p.J_Pc2_T12_.push_back(J_Pc2_T12);  // debug
        } else {
            J_Pc2_Twc2 = p.J_Pc2_T12_[0];
        }

        Eigen::Matrix<double, 3, 3> J_Pc2_Pw1;
        Eigen::Matrix<double, 3, 3> J_Pw1_Pc1;
        Eigen::Matrix<double, 3, 1> J_Pc1_z1;
        // TODO: 实现FEJ
        if (!onlyPoseUpdate_) {
            // Pc2 w.r.t Pc1 [3x3]
            J_Pc2_Pw1 = Tc2w.q_wb_.toRotationMatrix();

            // Pw1 w.r.t Pc1 [3x3]
            J_Pw1_Pc1 = p.host_->Twc_.q_wb_.toRotationMatrix();

            // Pc1 w.r.t z1 [3x1]
            const Eigen::Vector3d pc1Norm(p.GetPcNorm());
            J_Pc1_z1 << pc1Norm.x(), pc1Norm.y(), 1;
            // 使用逆深度表示
            const double d = pow(p.invZ_, 2);
            J_Pc1_z1 << -pc1Norm.x() / d, -pc1Norm.y() / d, -1 / d;
        }

        // 给整体雅可比矩阵赋值
        // H = J'*J, g = -J'*b;
        // 当前雅可比及梯度的行和列，用于构建上述H矩阵和g向量
        const int ai = j * resDim, aj = poseStartCol;
        const int bi = j * resDim, bj = pointStartCol + j * p.Size();
        // cout << "J_res_px2:\n" << J_res_px2 << endl;
        // cout << "J_px2_Pc2:\n" << J_px2_Pc2 << endl;
        // cout << "J_Pc2_T12:\n" << J_Pc2_T12 << endl;
        Eigen::Matrix<double, 2, 6> A =
            J_px2_Pc2 * J_Pc2_Twc2 * noUpdatePoseNum;
        double w = 1.0;  // 1.0 / lk1s[i]->invDepthCov_;
        H.block(aj, aj, A.cols(), A.cols()) += A.transpose() * A * w * rho[1];

        /******** -J.T * b的size为[J.cols() x 1]**************
            * | A.T  C.T  E.T |       | A.T*b1 + C.T*b2 + E.T*b3|
            * | B.T  D.T  F.T | * b = | B.T*b1 + D.T*b2 + F.T*b3|
            *
        *****************************************************/
        g.middleRows(aj, A.cols()) -= A.transpose() * r * w * rho[1];

        if (!onlyPoseUpdate_) {
            Eigen::MatrixXd B = J_px2_Pc2 * J_Pc2_Pw1 * J_Pw1_Pc1 * J_Pc1_z1;
            /******** 利用分块及稀疏矩阵性质直接计算H矩阵 ********
                * 否则，H=J.T * J由于没有利用到稀疏性，计算量将异常大
                * | A.T, C.T, E.T |   | A, B|
                * | B.T, D.T, F.T | * | C, D|
                *                     | E, F| = 
                * | A.T*A + C.T*C + E.T*E,  A.T*B + C.T*D + E.T*F |
                * | B.T*A + D.T*C + F.T*E,  B.T*B + D.T*D + F.T*F |
                * | J矩阵第1列相关项和， J矩阵第1列转置与第2列相关项和 |
                * | J矩阵第2列转置与第1列相关项和， J矩阵第2列相关项和 |
                * 观察D、E矩阵块的变化规律，可以写出如下的等式
            **************************************************/
            // TODO:添加胡伯核关于chi2的一阶导数
            H.block(aj, bj, A.cols(), B.cols()) +=
                A.transpose() * B * w * rho[1];
            H.block(bj, aj, B.cols(), A.cols()) +=
                B.transpose() * A * w * rho[1];
            H.block(bj, bj, B.cols(), B.cols()) +=
                B.transpose() * B * w * rho[1];
            g.middleRows(bj, B.cols()) -= B.transpose() * r * w * rho[1];
        }
    }
    cout << "jac info.all.cost: " << info.cost
         << " useful num: " << info.usefulNum << "\n";
    info.cost /= info.usefulNum;
    cout << "jac mean cost: " << info.cost << "\n";
    return info;
}

Eigen::VectorXd Optimizer::SchurCompleteSolve(
    const Eigen::MatrixXd& H, const Eigen::VectorXd& b, const int poseNum,
    const int pointNum, const int poseDim, const int pointDim) {
    /***
    *     T   p...
    * T   A   B
    * p   C   D
    * ...
    ***/
    // 对B进行边缘化，左乘形成上三角矩阵
    // | I          0 |   | A  B |   | A  B |
    // | -C*A.inv   I | * | C  D | = | 0  ΔA| ==> ΔA = -C*A.inv*B + D
    // 对C进行边缘化，右乘形成下三角矩阵
    // | A  B |   | I  -A.inv*B |   | A  0 |
    // | C  D | * | 0       I   | = | C  ΔA| = H' ==> 用来求pose
    // 可以得到:
    // | I        0 |   | A  B |   | I  -A.inv*B |   | A  0 |
    // |-C*A.inv  I | * | C  D | * | 0      I    | = | 0  ΔA| = H'

    // 自己可以构建舒尔补，左乘形成下三角矩阵，先求解pose增量，再求解point增量
    // | I  -B*D.inv |   | A  B |   | A-B*D.inv*C  0 |
    // | 0     I     | * | C  D | = |    C         D |
    // H -= lambdaMatrix;
    const int poseSize = poseNum * poseDim;
    const int pointSize = H.cols() - poseSize;

    const Eigen::MatrixXd& A = H.block(0, 0, poseSize, poseSize);
    const Eigen::MatrixXd& D =
        H.block(poseSize, poseSize, pointSize, pointSize);
    Eigen::MatrixXd Dinv = Eigen::MatrixXd::Zero(D.rows(), D.cols());
    chrono::steady_clock::time_point t0 = chrono::steady_clock::now();
    for (int i = 0; i < pointSize; i += pointDim) {
        //Dinv.block(i, i, pointDim, pointDim).noalias() = D.block(i, i, pointDim, pointDim).inverse();
        Dinv(i, i) = 1.0 / D(i, i);
    }
    Eigen::VectorXd deltaX = Eigen::VectorXd::Zero(poseSize + pointSize);
    if (A.isApproxToConstant(0)) {
        // 仅更新point
        deltaX.tail(pointSize) = Dinv * b.tail(pointSize);
        //cout << "D:\n"
        //     << D << endl
        //     << "Dinv:\n"
        //     << Dinv << endl
        //     << "b.tail(pointSize): " << b.tail(pointSize).transpose() << endl;
        //cout << setprecision(5)
        //     << "only update deltaPoint: " << deltaX.tail(pointSize).transpose()
        //     << endl;
        return deltaX;
    }
    const Eigen::MatrixXd& B = H.block(0, poseSize, poseSize, pointSize);
    const Eigen::MatrixXd& C = H.block(poseSize, 0, pointSize, poseSize);
    // cout << "B - C.T:\n" << B-C.transpose() <<endl;

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    //const Eigen::MatrixXd E = -B * Dinv;
    // E的计算耗时最长，利用Dinv是稀疏矩阵这一特性加速
    Eigen::MatrixXd E = Eigen::MatrixXd::Zero(B.rows(), Dinv.cols());
    for (int i = 0; i < B.rows(); ++i) {
        for (int j = 0; j < B.cols(); ++j)
            E(i, j) = -B(i, j) * Dinv(j, j);
    }
    chrono::steady_clock::time_point t1_1 = chrono::steady_clock::now();

    // 这是个稀疏矩阵，可以优化掉
    //Eigen::MatrixXd leftMatrix(H.rows(), H.cols());
    //leftMatrix.setIdentity();
    //leftMatrix.block(0, 0, poseSize, poseSize).setIdentity();
    //leftMatrix.block(0, poseSize, poseSize, pointSize).noalias() = E;
    //leftMatrix.block(poseSize, 0, pointSize, poseSize).setZero();
    //leftMatrix.block(poseSize, poseSize, pointSize, pointSize).setIdentity();
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();

    // 求pose增量
    //cout << "A:\n"
    //     << A << endl
    //     << "B:\n"
    //     << B << endl
    //     << "C:\n"
    //     << C << endl
    //     << "D:\n"
    //     << D << endl;
    //cout << "Dinv:\n" << Dinv << endl << "E:\n" << E << endl;
    Eigen::MatrixXd newA = A + E * C;
    //cout << "newA:\n" << newA << endl;
    // 根据leftMatrix矩阵的稀疏性，这里不需要其完整形式即可计算出new_b
    //Eigen::VectorXd new_b = leftMatrix * b;
    // | I  E|
    // | 0  I| * b
    Eigen::VectorXd new_b = b;
    new_b.head(poseSize) = b.head(poseSize) + E * b.tail(pointSize);
    Eigen::VectorXd deltaPose = newA.inverse() * (new_b).head(poseSize);
    chrono::steady_clock::time_point t3 = chrono::steady_clock::now();
    //cout << "b: " << b.transpose() << endl
    //     << "newb: " << new_b.transpose() << endl;
    // 求point增量
    // H * Δx = b ==> C*deltaX_pose + D*deltaX_point = b
    // D*deltaX_point = b - C*deltaX_pose
    // deltaX_point = D.inv * (b - C*deltaX_pose)
    Eigen::VectorXd deltaPoint = Dinv * (new_b.tail(pointSize) - C * deltaPose);
    chrono::steady_clock::time_point t4 = chrono::steady_clock::now();

    // cout << setprecision(5) << "deltaPoint: "<< deltaPoint.transpose() << endl;
    deltaX.head(poseSize) = deltaPose;
    deltaX.tail(pointSize) = deltaPoint;
    chrono::steady_clock::time_point t5 = chrono::steady_clock::now();

    cout << setprecision(5) << "deltaPose: " << deltaPose.transpose() << endl;
    cout << setprecision(5) << "deltaPoint: " << deltaPoint.transpose() << endl;

    cout << "calculate D.inv spend: "
         << to_string(chrono::duration<double>(t1 - t0).count()) << endl;
    cout << "calculate E mat spend: "
         << to_string(chrono::duration<double>(t1_1 - t1).count()) << endl;
    cout << "calculate left mat spend: "
         << to_string(chrono::duration<double>(t2 - t1_1).count()) << endl;
    cout << "calculate dPose spend: "
         << to_string(chrono::duration<double>(t3 - t2).count()) << endl;
    cout << "calculate dPoint spend: "
         << to_string(chrono::duration<double>(t4 - t3).count()) << endl;
    cout << "construct dX spend: "
         << to_string(chrono::duration<double>(t5 - t4).count()) << endl;
    return deltaX;
}

bool Optimizer::ExecuteWindowOptimize() {
    vector<Landmark*> debugAllConvergeLandmark = optLandmark_;
    ResidualInfo lastCost = CalculateResidualWindow(optLandmark_);

    // 只需要保留最老帧的信息即可，或者只固定首帧的pose进行优化在debug阶段也是可取的
    // 其信息已经通过深度点的传播转移到后面的KF中
    // MarginalizeOldestKeyFrame();

    // 如果是使用点-点匹配逻辑的话，那么应该先进行边缘化再转移点的控制权
    // 产生的问题是：那些没有host被边缘化，但是没有target的点不造成影响
    // 那些host被边缘化，但是仍有target的点，可能只剩一个target本身的观测
    RemoveOldestKeyFrame();
    //ShowPointCloud(debugAllConvergeLandmark, optLandmark_, "All vs Opt");

    // 丢失追踪，重新进行
    if (optLandmark_.size() < 100) {
        cerr << "optLandmark_ too small: " << optLandmark_.size() << endl;
        return false;
    } else {
        //cerr << "before opt local map" << endl;
        //ShowLocalMap(window_.back());
    }

    ResidualInfo firstCost = lastCost;
    bool status = false;
    chrono::steady_clock::time_point T1 = chrono::steady_clock::now();
    for (int i = 0; i < maxIte_; ++i) {
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

        lastCost = ConstructJ_H_b_g();

        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        cout << "ConstructJ_H_b_g spend: "
             << chrono::duration<double>(t2 - t1).count() << " sec." << endl;

        Eigen::VectorXd _lambda(H_.rows());
        _lambda.setConstant(lambda_);
        if (Hp_.rows() > 1) {
            if (config->messageLevel == MessageLevel::Debug) {
                cout << "Hp_: [" << Hp_.rows() << "x" << Hp_.cols() << "]"
                     << endl;
                cout << "g_p_: [" << g_p_.rows() << "x1]" << endl;
                cout << "H_: [" << H_.rows() << "x" << H_.cols() << "]" << endl;
                cout << "g_: [" << g_.rows() << "x1]" << endl;
            }
            cout << "Hp_[6x6]: " << setprecision(5)
                 << Hp_.diagonal().head(6).transpose() << endl;
            H_ += Hp_;
            g_ += g_p_;
            //cout << setprecision(5) << "Hp_: " << Hp_.diagonal().transpose() << endl;
            //cout << setprecision(5) << "g_p_: " << g_p_.transpose() << endl;
            cout << "Prior Message Added!!!" << endl;
            ;
        } else {
            _lambda.head(6).setConstant(DBL_MAX);  // 首帧的约束足够大
            cout << "Fixed First Frame!!!" << endl;
        }
        H_.diagonal() += _lambda;

        Eigen::VectorXd delta_x;
        if (!onlyPoseUpdate_) {
            chrono::steady_clock::time_point t3 = chrono::steady_clock::now();
            delta_x = SchurCompleteSolve(
                H_, g_, window_.size(), optLandmark_.size(),
                window_[0]->Twc_.Size(), optLandmark_[0]->Size());
            chrono::steady_clock::time_point t4 = chrono::steady_clock::now();
            cout << "SchurCompleteSolve spend: "
                 << chrono::duration<double>(t4 - t3).count() << " sec."
                 << endl;
        } else {
            delta_x = H_.colPivHouseholderQr().solve(g_);
        }
        // cout << setprecision(5) << "delta_x: " << delta_x.transpose() << endl;

        // 保留状态备份
        for (size_t i = 0; i < optLandmark_.size(); ++i) {
            optLandmark_[i]->CopyStatus();
        }

        for (size_t i = 0; i < window_.size(); ++i) {
            window_[i]->CopyStatus();
        }

        // 状态更新
        int updateId = 0;
        for (size_t i = 0; i < window_.size(); ++i) {
            const int startRow = i * window_[0]->Twc_.Size();
            window_[i]->Update(delta_x.middleRows(startRow, 3),
                               delta_x.middleRows(startRow + 3, 3));
        }
        if (!onlyPoseUpdate_) {
            updateId += window_.size() * window_[0]->Twc_.Size();
            for (size_t i = 0; i < optLandmark_.size(); ++i) {
                optLandmark_[i]->Update(
                    delta_x.middleRows(updateId, optLandmark_[0]->Size())[0]);
                ++updateId;
            }
        }

        // 判断当前更新是否有效，在使用新的pose计算cost时，可能会让一些点被设置为noUsed
        ResidualInfo newCost = CalculateResidualWindow(optLandmark_);
        if (newCost.usefulNum != lastCost.usefulNum && 0) {
            // TODO：这里需要使用更新前的pose
            cout << "[WARNING]: "
                 << "recalculate last cost, last usefulNUm, new usefulNum: "
                 << lastCost.cost << ", " << lastCost.usefulNum << ", "
                 << newCost.usefulNum << endl;
            // TODO: 重新计算时，需要使用旧的poses，旧的landmarks位置，以及新的landmarks的noUsed标志，比较麻烦
            lastCost = CalculateResidualWindow(optLandmark_, true);
            cout << "new usefulNum: " << lastCost.usefulNum << endl;
        }
        cout << fixed << "iterate " << i
             << " times, last_cost, new_cost: " << lastCost.cost << " "
             << newCost.cost << " lambda: " << lambda_ << "  usefulNum rate: "
             << double(lastCost.usefulNum) / optLandmark_.size() << endl;

        if (lambda_ != 0) {
            // 使用LM方法，考虑存在由于图像模糊投影不上的问题，因此newCost不能小于0
            if (lastCost.cost <= newCost.cost) {
                lambda_ *= 1.8;
                for (size_t i = 0; i < optLandmark_.size(); ++i) {
                    optLandmark_[i]->BackUpStatus();
                }
                for (size_t i = 0; i < window_.size(); ++i) {
                    window_[i]->BackUpStatus();
                }
            } else {
                lambda_ *= 0.3;
                lastCost = newCost;
                // 更新先验残差构成信息项
                if (g_p_.rows() > 1) {
                    cout << "delta_x: [" << delta_x.rows() << "x1]" << endl;
                    UpdatePriorConstraint(delta_x);
                }
            }
        }

        if (newCost.cost < 1e-9) {
            cout << "Congratulations! LM converge!!!" << endl;
            status = true;
            break;
        }
        if (lambda_ > 1e10) {
            cout << fixed << "lambad too large: " << lambda_ << endl;
            break;
        }
        chrono::steady_clock::time_point t5 = chrono::steady_clock::now();
        cout << "LM one iteration spend: "
             << chrono::duration<double>(t5 - t1).count() << " sec.\n"
             << endl;
    }
    chrono::steady_clock::time_point T2 = chrono::steady_clock::now();
    const double spendTime = chrono::duration<double>(T2 - T1).count();
    if (!status) {
        cerr << "Reach max iteration time or lambda too large" << endl;
    }
    cout << "First cost | final cost | decrease ratio: " << firstCost.cost
         << " | " << lastCost.cost << " | "
         << (lastCost.cost / firstCost.cost) * 100 << "%" << endl;
    cout << "Total Optimize spend " << spendTime << "s\n" << endl;

    //cerr << "after opt local map" << endl;
    //ShowLocalMap(window_.back());

    return status;
}

bool Optimizer::OptimizeCurFrame(KeyFrame::OpticalFlowStruct& optFlw,
                                 Pose& Twc2, const int curFid) {

    // 构建优化问题所需观测
    vector<Landmark*> lk1s;
    vector<Eigen::Vector2d> obvs;
    lk1s.reserve(optFlw.trackLandmark_.size());
    obvs.reserve(lk1s.size());
    constexpr int kDebugNum = 2000;
    for (size_t i = 0; i < optFlw.trackLandmark_.size(); ++i) {
        // TODO: FEJ指的是关于逆深度的线性化点在首次计算出逆深度值时
        Landmark* lk = optFlw.trackLandmark_[i];
        if (lk->initialized_) {
            lk->ResetFEJ();
            lk1s.emplace_back(lk);
            const cv::Point2f& p = optFlw.prevPts_[i];
            obvs.emplace_back(p.x, p.y);
#if defined(WRITE_MATCH_PAIR_IMAGE)
            //DrawProjectCase(*lk, obvs.back(), optFlw.prevImg_, Twc2);
#endif
        }
        if (lk1s.size() > kDebugNum) {
            break;
        }
    }

    for (size_t i = 0; i < optFlw.trackHistoryLandmark_.size(); ++i) {
        // TODO: FEJ指的是关于逆深度的线性化点在首次计算出逆深度值时
        Landmark* lk = optFlw.trackHistoryLandmark_[i];
        if (lk->initialized_) {
            lk->ResetFEJ();
            lk1s.emplace_back(lk);
            const cv::Point2f& p = optFlw.prevHistoryPts_[i];
            obvs.emplace_back(p.x, p.y);
#if defined(WRITE_MATCH_PAIR_IMAGE)
            //DrawProjectCase(*lk, obvs.back(), optFlw.prevImg_, Twc2);
#endif
        }

        if (lk1s.size() > kDebugNum) {
            break;
        }
    }

    if (lk1s.empty()) {
        cout << "useful landmark num for opt is: " << lk1s.size()
             << " Error! may be no initialized???\n";
        //exit(-1);
        return false;
    }

    cout << "use " << lk1s.size() << " landmarks to optimize!\n";
#if defined(WRITE_MATCH_PAIR_IMAGE)
    //WriteDebugTriangulateCase2Video(curFid);
#endif
    bool status = false;
    double initLambda = lambda_;
    lambda_ = initLambda;
    ResidualInfo lastCost =
        CalculateResidualCurFrame(lk1s, obvs, Twc2, optFlw.prevImg_, true);
    ResidualInfo firstCost = lastCost;

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    for (int i = 0; i < maxIte_; ++i) {
        lastCost = CalculateJacobianAndCostCurFrame(lk1s, obvs, Twc2, H_, g_);
        Eigen::VectorXd _lambda(H_.rows());
        _lambda.setConstant(lambda_);
        if (H_.diagonal().head(6).isApproxToConstant(0)) {
            _lambda.head(6).setConstant(0);
        }

        // debug, 返回J, 判断H, g计算的正确性
        // Eigen::MatrixXd _H = J.transpose() * J;
        // Eigen::VectorXd _g = -J.transpose() * b;
        // const Eigen::MatrixXd dH = H-_H;
        // const Eigen::VectorXd dg = g-_g;
        // cout << setprecision(5) << "H-_H:\n " << dH.diagonal().transpose() << endl << endl;
        // cout << setprecision(5) << "g-_g:\n " << dg.transpose() << endl << endl;
        // cout << "dH, dg norm: " << dH.norm() << " " << dg.norm() << endl;
        // H = _H;
        // g = _g;

        H_.diagonal() += _lambda;
        //cout << setprecision(5) << "H_:\n " << H_.diagonal().transpose() << endl
        //     << endl;
        //cout << setprecision(5) << "g_:\n " << g_.transpose() << endl << endl;
        Eigen::VectorXd delta_x;
        // H_ /= lastCost.usefulNum;
        // g_ /= lastCost.usefulNum; 不需要除以吧
        if (!onlyPoseUpdate_) {
            delta_x = SchurCompleteSolve(H_, g_, 1, lk1s.size(), Twc2.Size(),
                                         lk1s[0]->Size());
        } else {
            // delta_x = H_.colPivHouseholderQr().solve(g_);
            // delta_x = H_.inverse() * g_;
            delta_x = H_.ldlt().solve(g_);
            cout << setprecision(5) << "delta_pose: " << delta_x.transpose()
                 << endl;
        }
        // cout << setprecision(5) << "delta_x: " << delta_x.transpose() << endl;

        for (size_t i = 0; i < lk1s.size() && !onlyPoseUpdate_; ++i) {
            lk1s[i]->CopyStatus();
        }
        const Pose poseBackup = Twc2;

        // 状态更新
        int updateId = 0;
        const int startRow = 0;
        Twc2.Update(delta_x.middleRows(startRow, 3),
                    delta_x.middleRows(startRow + 3, 3));

        if (!onlyPoseUpdate_) {
            updateId += Twc2.Size();
            for (size_t i = 0; i < lk1s.size(); ++i) {
                // TODO：需要考虑lk1s[i]被设置为不使用的情况，
                // 由于有匹配特征点，因此这里暂不设置为不使用
                lk1s[i]->Update(
                    delta_x.middleRows(updateId, lk1s[0]->Size())[0]);
                ++updateId;
            }
        }

        // 判断当前更新是否有效
        ResidualInfo newCost =
            CalculateResidualCurFrame(lk1s, obvs, Twc2, optFlw.prevImg_);
        constexpr int maxCullPointEachTime = 10;
        if (abs(lastCost.usefulNum - newCost.usefulNum) >
            maxCullPointEachTime) {
            // TODO: 这里暂时不考虑深度值也更新的情况
            // 需要使用旧的poses，旧的landmarks位置，以及新的landmarks的noUsed标志，比较麻烦
            // 但由于这里只更新pose，所以影响应该不大吧！！！
            cout << "[WARNING-0]: "
                 << "recalculate last cost, last usefulNUm, new usefulNum: "
                 << lastCost.cost << ", " << lastCost.usefulNum << ", "
                 << newCost.usefulNum << endl;
            lastCost = CalculateResidualCurFrame(lk1s, obvs, poseBackup,
                                                 optFlw.prevImg_);
            cout << "new usefulNum: " << lastCost.usefulNum << endl;
        }
        // else if(lastCost.usefulNum - newCost.usefulNum >= maxCullPointEachTime) {
        //     // 避免一次删除过多point
        //     newCost = lastCost;
        //     lambda_ = 1e21;
        // }
        cout << fixed << "iterate " << i
             << " times, lastCost | newCost: " << lastCost.cost << " | "
             << newCost.cost << " &lambda: " << lambda_
             << " &usefulNum rate: " << double(lastCost.usefulNum) / lk1s.size()
             << endl
             << endl;

        if (lambda_ != 0) {
            // LM 方法
            if (lastCost.cost <= newCost.cost) {
                lambda_ *= 1.8;
                for (size_t i = 0; i < lk1s.size() && !onlyPoseUpdate_; ++i) {
                    lk1s[i]->BackUpStatus();
                }
                Twc2 = poseBackup;
            } else if (lastCost.cost > newCost.cost) {
                // 状态量已在上面更新
                lambda_ *= 0.3;
                lastCost = newCost;
            }
        }
        if (newCost.cost < 1e-9) {
            cout << "Congratulations! LM or GN converge!!!" << endl;
            status = true;
            break;
        }
        if (lambda_ > 1e20) {
            cout << fixed << "lambad too large: " << lambda_ << endl;
            break;
        }
    }
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    const double spendTime = chrono::duration<double>(t2 - t1).count();
    if (!status) {
        cerr << "Reach max iteration time or lambda too large" << endl;
    }
    cout << "First cost | final cost | decrease ratio | usefulNum ratio: "
         << firstCost.cost << " | " << lastCost.cost << " | "
         << (1. - lastCost.cost / firstCost.cost) * 100 << "%"
         << " | " << (double(lastCost.usefulNum) / lk1s.size()) * 100 << "%"
         << endl;
    cout << "Total Optimize spend " << spendTime << "s\n" << endl;

    return status;
}

void Optimizer::AddOneKeyFeame(KeyFrame* kf) {
    // TODO: 上一帧需要添加对当前KF的观测，此外，
    // 如果上一帧的landmark继续被下一帧看到，应该也要实现持续跟踪的效果
    // 考虑是否需要转移landmark所有权？
    // 直接在新KF中提取关键点，并放入optLkw_结构中，同时保留上一KF的跟踪结果仍进行跟踪
    KeyFrame* lastKf = window_.empty() ? nullptr : window_.back();
    kf->InitializeLandmark(lastKf);
    int historyTriSucceedNum = 0;
    int prevTriSucceedNum = 0;
    if (!window_.empty()) {
        unordered_map<KeyFrame*, Pose> kf2T12;
        KeyFrame::OpticalFlowStruct optLkw = window_.back()->optFlw_;
        for (size_t i = 0; i < optLkw.trackHistoryLandmark_.size(); ++i) {
            Landmark* lk = optLkw.trackHistoryLandmark_[i];
            // 仍被当前帧观测到，可以进行深度滤波更新，或者进行多视角优化
            const cv::Point2f& p = optLkw.prevHistoryPts_[i];  // curFrameObv
            const Eigen::Vector2d curObv(p.x, p.y);
            lk->target_.insert({kf, curObv});
            if (!lk->initialized_) {
                // 三角化
                double idepth1 = 0;
                if (!kf2T12.count(lk->host_)) {
                    kf2T12[lk->host_] = lk->host_->Tcw_ * kf->Twc_;
                }
                const Pose& T12 = kf2T12.at(lk->host_);
                if (config->useDepthImage ||
                    !GetHostFrameObservationInvDepth(
                        lk->uv_, curObv, cam_->Kinv_[0], T12, idepth1)) {
                    continue;
                }
                //const double var = CalculateVarianceByOffsetPx2(
                //    lk->uv_, curObv, Eigen::Vector2d(2, 2), cam_->Kinv_[0], T12,
                //    1.0);
                //lk->ObvUpdate(idepth1, var);
                lk->SetTriangulateResult(idepth1);
                ++historyTriSucceedNum;
#if defined(WRITE_MATCH_PAIR_IMAGE)
//DrawTriangulateCase(idepth1, *lk, curObv.cast<int>(),
//                    kf->debugGrayImg_, T12);
#endif
            }
        }

        for (size_t i = 0; i < optLkw.prevPts_.size(); ++i) {
            Landmark* lk = optLkw.trackLandmark_[i];
            const cv::Point2f& p = optLkw.prevPts_[i];  // curFrameObv
            const Eigen::Vector2d curObv(p.x, p.y);
            lk->target_.insert({kf, curObv});
            // 仅在这里三角化一次
            if (!lk->initialized_) {
                double idepth1 = 0;
                if (!kf2T12.count(lk->host_)) {
                    kf2T12[lk->host_] = lk->host_->Tcw_ * kf->Twc_;
                }
                const Pose& T12 = kf2T12.at(lk->host_);

                if (config->useDepthImage ||
                    !GetHostFrameObservationInvDepth(
                        lk->uv_, curObv, cam_->Kinv_[0], T12, idepth1)) {
                    continue;
                }
                // TODO：这里如何三角化两帧，因为事实上optLkw仅记录了之前Landmark，
                // 难道只能使用上上关键帧来进行跟踪？或者直接在这里原地创建kf2的关键点！！！
                // KF2新创建的landmark在不是历史帧的观测时，事实上也无法三角化，而能够三角化的只需要由历史帧来跟踪即可
                // 因此创建新关键帧的条件应该是跟踪到的历史Landmark数量？？？
                // 事实上，我们只需要优化当前帧的Twc即可利用历史KF的“已初始化”地图点进行BA优化，
                // 因此就不需要在当前关键帧kf2保留idepth2

                //const double var = CalculateVarianceByOffsetPx2(
                //    lk->uv_, curObv, Eigen::Vector2d(2, 2), cam_->Kinv_[0], T12,
                //    1.0);
                //lk->ObvUpdate(idepth1, var);
#if defined(WRITE_MATCH_PAIR_IMAGE)
                //DrawTriangulateCase(idepth1, *lk, curObv.cast<int>(),
                //                    kf->debugGrayImg_, T12);
#endif
                lk->SetTriangulateResult(idepth1);
                ++prevTriSucceedNum;
            }
        }

#if defined(WRITE_MATCH_PAIR_IMAGE)
        WriteDebugTriangulateCase2Video(kf->id_);
#endif

        // 这里的lastKf肯定不为nullptr，但是存在着三角化数量不够的情况
        //for (Landmark* lk : lastKf->landmark_) {
        //    if (lk != nullptr && lk->ManySupport() && lk->Converge()) {
        //        interaction->allMapPoints.push_back(lk->GetPw());
        //    }
        //}
        // RemoveOneKeyframe(*kf); 在局部BA优化过程中移除
        // 同时需要把所有由该关键帧首次观测到的地图点转移所有权
        // 并且删除其余地图点在该帧上的观测
    } else {
        // 创建的是首帧关键帧，是否需要赋值prevHistoryPts_？
        // 应该是不需要的
    }
    // TODO 1：进行滑窗BA
    // TODO 2：进行三角化工作

    window_.push_back(kf);
    cout << "add kf id: " << kf->id_ << "\n";
    cout << fmt::format(
        "Triangulate by KF_{} report: trackHistoryLandmark size: {}, "
        "historyTriSucceedNum: {}, "
        "prev trackLandmark size: {}, prevTriSucceedNum: {}\n",
        kf->id_, window_.back()->optFlw_.trackHistoryLandmark_.size(),
        historyTriSucceedNum, window_.back()->optFlw_.trackLandmark_.size(),
        prevTriSucceedNum);
}

void Optimizer::WriteDebugTriangulateCase2Video(const int curFid) {
    cout << "triPointMapDebugImage_ size: " << triPointMapDebugImage_.size()
         << "\n";
    if (triPointMapDebugImage_.empty()) {
        return;
    }

    // 初始化边缘匹配的debug视频写入器
    string videoPath = config->debugMessageSaveFolder;

    const string curVideoPath =
        fmt::format("{}/kf_id_{}_f_id_{}_triangulate.avi", videoPath,
                    window_.back()->id_, curFid);
    int fourcc = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
    int fps = 30;
    debugTriangulateWriter_.open(
        curVideoPath, fourcc, fps,
        triPointMapDebugImage_.begin()->second[0].size(), true);
    if (!debugTriangulateWriter_.isOpened()) {
        cerr << "Open debug video path: " << curVideoPath << " failed" << endl;
        //exit(-1);
        return;
    }
    cout << "Open debug video path: " << curVideoPath << endl;
    for (auto& nameMapImgs : triPointMapDebugImage_) {
        const string pointName = nameMapImgs.first;
        for (cv::Mat& img : nameMapImgs.second) {
            debugTriangulateWriter_.write(img);
            img.release();
            cout << "write tri img name: " << pointName << "\n";
        }
    }
    debugTriangulateWriter_.release();
    triPointMapDebugImage_.clear();
}

void Optimizer::RemoveOldestKeyFrame() {
    if (window_.size() <= config->maxKFnumInWindow) {
        return;
    }
    cout << "window size: " << window_.size() << " begin remove oldest" << endl;
    KeyFrame* oldest = window_[0];
    // TODO：比较倒数第3帧，以选择是否删除最老帧
    window_.erase(window_.begin());
    cout << "window[0]: " << window_[0] << endl;
    // 释放KF，其对应的landmark已经释放
    cout << "[WARNING] kf: " << oldest << " Set out of range flag" << endl;
    // delete oldest; // 不能直接释放，因为其余指向该位置的指针并不会变成nullptr
    oldest->SetOutOfRange();
    historicalKF_.push_back(oldest);  // TODO: 排查内存泄漏，关闭
    cout << "historicalKF_.size: " << historicalKF_.size() << endl;

    for (int i = static_cast<int>(window_.size() - 1); i >= 0; ++i) {
        KeyFrame* kf = window_[i];
        if (kf == oldest) {
            continue;
        }

        vector<Landmark*> lks = kf->landmark_;
        for (Landmark* lk : lks) {
            if (lk->host_ == oldest && lk->target_.count(kf)) {
                // 转移地图点所有权
                lk->TransformHost2OtherKF(kf);
            }
        }
    }

    oldest->ReleaseMat();
    // 删除老帧看看是否会有影响
    // delete oldest;
    return;
}

bool Optimizer::SetOptimizeVariables() {
    if (window_.size() < 2) {
        cerr << "Window size: " << window_.size() << " < 2" << endl;
        return false;
    }

    // 添加有效地图点进行优化
    set<Landmark*> ps;
    // 最新帧作为量测帧来更新已有KF的深度
    for (int i = 0; i < window_.size() - 1; ++i) {
        KeyFrame* kf = window_[i];
        vector<Landmark*>& ld = kf->landmark_;
        for (Landmark* p : ld) {
            // 重置FEJ保存的雅可比
            p->ResetFEJ();
            if (p != nullptr && !ps.count(p) && !p->IsOutOfRange() &&
                p->Converge()) {
                ps.insert(p);
            }
        }
    }
    cout << ps.size() << " Landmarks and " << window_.size()
         << " KFs will participate optimization!!!" << endl;
    optLandmark_.clear();
    optLandmark_ = vector<Landmark*>(ps.begin(), ps.end());
    // 按地址从小到大排序
    //sort(optLandmark_.begin(), optLandmark_.end(), [](Landmark *p1, Landmark*p2){return p1 < p2;});
    return optLandmark_.size() > 100;
}

int Optimizer::SampleUsefulLandmark() {
    optLandmark_.clear();
    for (int i = 0; i < static_cast<int>(window_.size()) - 1; ++i) {
        for (Landmark* p : window_[i]->landmark_) {
            // 地图点有被其他关键帧看到
            if (p->target_.empty()) {
                continue;
            }
            optLandmark_.emplace_back(p);
        }
    }

    return optLandmark_.size();
}

Optimizer::ResidualInfo Optimizer::CalculateResidualWindow(
    const std::vector<Landmark*>& optLandmark, const bool useBackUpStatus) {
    ResidualInfo info;

    for (size_t i = 0; i < optLandmark.size(); ++i) {
        Landmark* p = optLandmark[i];
        if (p->noUsed_) {
            continue;
        }
        KeyFrame* host = p->host_;
        const Eigen::Vector3d pc1 = p->GetPc(useBackUpStatus);
        const Eigen::Vector3d pw =
            useBackUpStatus ? host->TwcBack_ * pc1 : host->Twc_ * pc1;

        for (const auto& kf2obv : p->target_) {
            KeyFrame* target = kf2obv.first;
            if (target == host) {
                continue;
            }

            const Eigen::Vector2d& obv = kf2obv.second;
            const Eigen::Vector3d pc2 =
                useBackUpStatus ? target->TcwBack_ * pw : target->Tcw_ * pw;
            const Eigen::Vector2d px2 = cam_->Project2PixelPlane(pc2);

            const bool inRange = InRange(target->grayImg_, px2.cast<int>());
            if (!inRange) {
                // 需要认识到的是，未更新前的{poses, landmarks}一定是可以投影成功的！！！
                // 所以理论上，这一步只会被更新后的{poses, landmarks}执行
                p->noUsed_ = true;
                continue;
            } else {
                Eigen::Vector2d r = px2 - obv;
                double chi2 = r.squaredNorm();
                Eigen::Vector2d rho;
                HuberLoss(chi2, rho);
                info.cost += rho[0];
                ++info.usefulNum;
            }
        }
    }

    info.cost /= info.usefulNum;
    if (isnan(info.cost) || isinf(info.cost)) {
        info.cost = DBL_MAX;
    }
    return info;
}

// TODO： 先不考虑边缘化，而是直接丢弃首帧
void Optimizer::MarginalizeOldestKeyFrame() {
    if (static_cast<int>(window_.size()) <= config->maxKFnumInWindow) {
        return;
    }
    Hp_.resize(1, 1);
    g_p_.resize(1, 1);
    /*********************************************************
    * 注意：VINS-MONO论文中的r_p, Hp分别代表先验残差、先验雅可比，
    * 即 先验约束项 |r_p - Hp * X|^2 <==> |r_p - Jp * X|^2
    * 注意：VINS-MONO论文中，多处出现H矩阵，其均不代表J’*J!!!，而是代表J
    * 参考为：https://github.com/StevenCui/VIO-Doc/tree/master
    ***********************************************************
    * |A B|   |dx1|   |g1|
    * |C D| * |dx2| = |g2| ==>
    * 将A边缘化掉，得：
    * |E F|   |dx1|   |h1|
    * |0 G| * |dx2| = |h2| ==>
    * G*dx2 = h2, 并且，dx2满足：
    * {我们知道，对于一个线性化的量测方程而言，有：
    * J'*J * dx = -J' * b，
    ******************************************************
    * 令G = J' * J, h2 = -J' * b, 
    * 那么构建先验约束： r = |b - J*X|^2，该式在求解极小值点dx的过程中，
    * 恰好能满足出现： G*dx = h2这一先验约束
    *********************************************************
    * 所以，只被margTwc观测到的地图点，我们不再用它构建方程，直接丢弃
    * 既被margTwc又被其他Twc观测到的地图点，不将其边缘化，而是继续更新
    *******************************************************/

    // Stpe: 首先将需要被边缘化的地图点的行和列移动到左上角
    /****************************************************************
    *       T0    T1    d0    d1                       T0   d1   T1   d0
    * T0  T0T0  T1T0  doT0  d1T0                  T0 T0T0 d1T0 T1T0 d0T0
    * T1  T0T1  T1T1  d0T1  d1T1 ==> 移动d1到左上角 d1 T0d1 d1d1 T1d1 d0d1
    * d0  T0d0  T1d0  d0d0  d1d0                  T1 T0T1 d1T1 T1T1 d0T1
    * d1  T0d1  T1d1  d0d1  d1d1                  d0 T0d0 d1d0 T1d0 d0d0
    * 直接操作信息矩阵H_来移动看起来是不可能的，但是我们可以先组织优化变量的顺序，
    * 再计算排序后的H_矩阵，这样直接边缘化左上角就简单了
    *****************************************************************/
    set<Landmark*> margLandmark;
    // TODO: 选择另一种策略移除一帧，类似Landmark的处理方式，将其移到window_[0]再构建信息矩阵H即可
    KeyFrame* margKF = window_[0];
    for (int i = 0; i < optLandmark_.size(); ++i) {
        Landmark* p = optLandmark_[i];
#ifndef USE_DT_RESIDUAL
        if (p->target_.empty() ||
            (p->target_.size() == 1 && p->target_.count(margKF))) {
#else
        if (p->host_->id_ == margKF->id_) {
#endif
            margLandmark.insert(p);
        }
    }

    vector<Landmark*> sortMargOptLandmark;
    for (Landmark* p : margLandmark) {
        // OK, 这样就实现了将边缘化地图点移到左上角的目的啦！！！
        sortMargOptLandmark.push_back(p);
    }
    for (Landmark* p : optLandmark_) {
        if (!margLandmark.count(p)) {
            sortMargOptLandmark.push_back(p);
        }
    }
    // 根据新的Landmark顺序，构建信息矩阵H_，并保存FEJ
    optLandmark_ = sortMargOptLandmark;
    cout << "total, marg, left landmars: " << optLandmark_.size() << " "
         << margLandmark.size() << " "
         << (optLandmark_.size() - margLandmark.size()) << endl;
    // 这里可以实现将线性化点固定在Marginalization时刻
#ifndef USE_DT_RESIDUAL
    ConstructJ_H_b_g_byMatch();
#else
    ConstructJ_H_b_g();
#endif

    //ShowPointCloud(sortMargOptLandmark, optLandmark_, "Marg left landmark");

    // Step:接下来计算相关先验Hp, g_p
    const int poseDim = window_[0]->Twc_.Size();
    const int depthDim = 1;
    //const int margDim = poseDim + margLandmark.size() * depthDim;
    // 这里我们直接将最老帧的landmark丢弃不用，只保留边缘化最老帧的信息
    const int margDim = poseDim;
    const int leftDim = H_.cols() - margDim;
    // 使用舒尔补进行边缘化H矩阵，并形成上三角矩阵
    // | I          0 |   | A  B |   | A  B |
    // | -C*A.inv   I | * | C  D | = | 0  ΔA| ==> ΔA = -C*A.inv*B + D
    Eigen::MatrixXd A = H_.block(0, 0, margDim, margDim);
    if (A.diagonal().squaredNorm() < 1.) {
        Hp_.resize(1, 1);
        g_p_.resize(1, 1);
        return;
    }
    Eigen::VectorXd eps(margDim);
    // To avoid A is all Zero，对角线的约束照例说也不应该为0
    eps.setConstant(0);
    A.diagonal() += eps;
    const Eigen::MatrixXd& B = H_.block(0, margDim, margDim, leftDim);
    const Eigen::MatrixXd& C = H_.block(margDim, 0, leftDim, margDim);
    const Eigen::MatrixXd& D = H_.block(margDim, margDim, leftDim, leftDim);
    // TODO: 当边缘化landmark时，可以使用稀疏性求逆
    const Eigen::MatrixXd invA = A.inverse();
    const Eigen::MatrixXd temp = -C * invA;
    Hp_ = -temp * B + D;
    cout << "debug A: " << setprecision(5) << A.diagonal().transpose() << endl
         << "debug B: " << B.diagonal().transpose() << endl
         << "debug D: " << D.diagonal().head(12).transpose() << endl
         << "debug invA: " << invA.diagonal().transpose() << endl
         << "debug temp: " << temp.diagonal().transpose() << endl;
    // | A  B  |   |x1|   | I          0 |   |g1|
    // | 0  ΔA | * |x2| = | -C*A.inv   I | * |g2| ==>
    // TODO: 留下来的状态量X2如果更新，右边的先验残差怎么变呢?
    g_p_ = temp * g_.head(margDim) + g_.tail(leftDim);

    // 虽然我们只保留了最老帧的信息，但是这里仍要从optLandmark中删除掉以最老帧为host的landmark
    optLandmark_.erase(optLandmark_.begin(),
                       optLandmark_.begin() + margLandmark.size());

    // 易知，先验残差为： |Jp*X - b_p|^2. 其中，Hp_=Jp'*Jp，因此可以得到Jp，g_p_=Jp'*b_p，因此可以得到先验残差b_p(VINS-MONO)
    // 根据G-N方法，展开先验残差项得:
    // (Jp*X)'*(Jp*x) - 2*(Jp*X)'*b_p + b_p'*b_p ==>
    // X'*Jp'*Jp*X - 2*X'*Jp'*b_p + b_p'*bp ==> 极小值点在该式导数为0处，即解满足:
    // Jp'*Jp*X = 2Jp'*bp
    // 如果令 r_p = Jp*Xmarg - b_p，那么当Xnew = Xmarg+ΔX时，先验残差更新为：
    // r_p += Jp*ΔX
    //
    //
    // 事实上，上式理解为：
    // Hp_*(X-Xmarg) = g_p_, 那么，当Xnew=X+ΔX后，有==>
    // Hp_*(X-Xmarg+ΔX) = g_p_ + Hp_*ΔX，因此，当X更新后，我们需要同步更新
    // g_p_ += Hp_*ΔX

    // 暂时不边缘化点，而是直接丢弃
    //if(!margLandmark.empty()) {
    //    // 构建完H矩阵后，可以从优化地图点中移除marg landmark
    //    optLandmark_.erase(optLandmark_.begin(), optLandmark_.begin() + margLandmark.size());
    //}
}

Optimizer::ResidualInfo Optimizer::ConstructJ_H_b_g() {
    // 构建H, g
    // 给出每个KF对应的在H矩阵中的位置
    map<const KeyFrame*, int> kfMapCol;
    map<const KeyFrame*, int> debugKFMapResidualNum;
    for (size_t i = 0; i < window_.size(); ++i) {
        kfMapCol.insert({window_[i], i * 6});
        debugKFMapResidualNum.insert(
            {window_[i], 0});  // 统计每个图像对应的residual数量
    }
    const int poseDim = window_[0]->Twc_.Size();
    const int depthDim = 1;

    // 或许我们不知道residual，Jacobian的行数，但是H矩阵以及g向量的维度是可知的
    const int variableDim =
        window_.size() * poseDim + optLandmark_.size() * depthDim;
    cout << "opt variable dim: " << variableDim << endl;
    H_.resize(variableDim, variableDim);
    H_.setZero();
    g_.resize(variableDim);
    g_.setZero();
    int noInrangeNum = 0;
    int depthErrorNum = 0;
    int targetErrorNum = 0;
    ResidualInfo info;
    // clang-format off
    // 计算residual & jacobian
    /*********
    *    T0 T1 ... d0 d1 ...
    * r0
    *********/
    // clang-format on
    const int depthStartCol = window_.size() * poseDim;
    const int resDim = 2;
    int resNum = 0;  // 显示当前计算到雅可比的第几行
    // TODO: 有很多landmark不能参与计算，需要将其排除在H及g信息之外，典型的如：
    // opt variable dim: 22902
    // usefulNum, depthErrorNum, targetErrorNum, noInrangeNum, allBadNum: 804 0 23938 1 23939
    for (size_t i = 0; i < optLandmark_.size(); ++i) {
        Landmark* p = optLandmark_[i];
        KeyFrame* host = p->host_;
        const Eigen::Vector3d pc1 = p->GetPc();
        const Eigen::Vector3d pw = host->Twc_ * pc1;

        // res w.r.t (u2, v2) [2x2]的单位矩阵
        // px2 w.r.t Pc2 [2x3]
        const Eigen::Matrix<double, 2, 3> J_px2_Pc2Norm =
            p->cam_->K_[0].block(0, 0, 2, 3);

        // 最新关键帧没有反向追踪能力
        if (host->id_ != window_.back()->id_) {
            // 我们把所有帧上的深度图投影到最新帧，并优化滑窗内的所有pose
            for (const auto& kf2obv : p->target_) {
                // TODO：需要注意每个关键帧、每个landmark在H矩阵中的位置
                KeyFrame* target = kf2obv.first;

                const Eigen::Vector3d pc2 = target->Tcw_ * pw;
                const Eigen::Vector2d px2 = cam_->Project2PixelPlane(pc2);
                if (!InRange(target->grayImg_, px2.cast<int>())) {
                    p->noUsed_ = true;
                    ++noInrangeNum;
                    continue;
                }

                // 经过校验，可以构建residual和jacobian
                const Eigen::Vector2d r = px2 - kf2obv.second;

                // 使用胡伯核函数剔除异常残差值
                double chi2 = r.squaredNorm();
                Eigen::Vector2d rho;
                HuberLoss(chi2, rho);
                info.cost += rho[0];
                ++info.usefulNum;

                debugKFMapResidualNum[target] += 1;
                resNum += resDim;

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

                Eigen::Matrix<double, 3, 3> J_Pc2Norm_Pc2;
                const double d = 1 / pc2.z();
                const double d2 = 1. / pow(pc2.z(), 2);
                J_Pc2Norm_Pc2 << d, 0, -pc2.x() * d2, 0, d, -pc2.y() * d2, 0, 0,
                    0;
                const Eigen::Matrix<double, 2, 3> J_px2_Pc2 =
                    J_px2_Pc2Norm * J_Pc2Norm_Pc2;
                const Eigen::Matrix<double, 2, 3>& J_res_Pc2 = J_px2_Pc2;

                // FEJ
                //if (!p->J_Pc2_Twc2.count(target)) {
                // if(!p->J_Pc2_Twc2.count(target) || 1) {
                // Pc2 w.r.t Twc2 : Pc2 = Twc2.inv * Pw
                Eigen::Matrix<double, 3, 6>
                    J_Pc2_Twc2;  // ------------------------> optimization variable
                const Eigen::Vector3d dt = pw - target->Twc_.t_wb_;
                // Pc2 w.r.t Rwc2
                J_Pc2_Twc2.block(0, 0, 3, 3) = SkewSymmetric(
                    target->Tcw_.q_wb_ * dt);  // Twc_.q_wb_.inverse()
                // Pc2 w.r.t Pwc2
                J_Pc2_Twc2.block(0, 3, 3, 3) =
                    -target->Tcw_.q_wb_
                         .toRotationMatrix();  // Twc.q_wb.R.transpose()

                // Pc2 w.r.t Pw
                // TODO: 这里也应该要使用首次的Tcw值吧！！！由于target有多帧，所以要保留多个
                const Eigen::Matrix3d J_Pc2_Pw =
                    target->Tcw_.q_wb_.toRotationMatrix();

                //if (p->J_Pc2_Pw.count(target)) {
                //    // just for debug
                //    // 暂时不考虑首次雅可比
                //    p->J_Pc2_Twc2[target] = J_Pc2_Twc2;
                //    p->J_Pc2_Pw[target] = J_Pc2_Pw;
                //} else {
                //    // p->J_Pc2_Twc2.insert({target, J_Pc2_Twc2});
                //    // p->J_Pc2_Pw.insert({target, J_Pc2_Pw});
                //}

                //if (p->J_Pw_z.empty()) {
                // if(p->J_Pw_z.empty() || 1) {
                // Pw w.r.t Twc1 : Pw = Twc1 * Pc1 = Rwc1 * pc1 + Pwc1
                Eigen::Matrix<double, 3, 6>
                    J_Pw_Twc1;  // ------------------------> optimization variable
                // Pw w.r.t Rwc1
                J_Pw_Twc1.block(0, 0, 3, 3) =
                    -host->Twc_.q_wb_.toRotationMatrix() * SkewSymmetric(pc1);
                // Pw w.r.t Pwc1
                J_Pw_Twc1.block(0, 3, 3, 3) = Eigen::Matrix3d::Identity();

                // Pw w.r.t Pc1
                const Eigen::Matrix3d J_Pw_Pc1 =
                    host->Twc_.q_wb_.toRotationMatrix();

                // Pc1 w.r.t z
                const Eigen::Vector3d pc1Norm(p->GetPcNorm());
                Eigen::Vector3d J_Pc1_z{pc1Norm.x(), pc1Norm.y(),
                                        1};  // --------> optimization variable
                                             // 使用逆深度表示
                const double d1 = pow(p->invZ_, 2);
                J_Pc1_z << -pc1Norm.x() / d1, -pc1Norm.y() / d1, -1 / d1;

                const Eigen::Matrix<double, 3, 1> J_Pw_z = J_Pw_Pc1 * J_Pc1_z;

                // 不考虑使用首次雅可比
                //if (!p->J_Pw_z.empty()) {
                //    // just for debug
                //    p->J_Pw_z.clear();
                //    p->J_Pw_Twc1.clear();
                //}
                //p->J_Pw_z.push_back(J_Pw_Pc1 * J_Pc1_z);
                //p->J_Pw_Twc1.push_back(J_Pw_Twc1);
                //}
                //}

                // Residual w.r.t optimization variables Jacobian
                Eigen::Matrix<double, 2, 6> A1 =
                    J_res_Pc2 * J_Pc2_Pw * J_Pw_Twc1;  // J_res_Pw * J_Pw_Twc1;
                if (host == window_[0]) {
                    // fixed滑动窗口第一帧
                    //A1.setZero();
                }
                Eigen::Matrix<double, 2, 6> A2 =
                    J_res_Pc2 * J_Pc2_Twc2;  // J_res_Pc2 * J_Pc2_Twc2;
                if (target == window_[0]) {
                    //A2.setZero();
                }
                const Eigen::Matrix<double, 2, 1> B =
                    J_res_Pc2 * J_Pc2_Pw *
                    J_Pw_z;  // J_res_Pw * J_Pw_Pc1 * J_Pc1_z;

                const double w = 1.0;  // /p->depthCov_;
                const int a1i = resNum, a1j = kfMapCol[host], a2i = resNum,
                          a2j = kfMapCol[target], bi = resNum,
                          bj = depthStartCol + i;
                // clang-format off
            /********************* 利用稀疏性计算H=J'*J ****************************
            * | A1'|
            * | A2'| * | A1 A2 B |
            * | B' |
            * =
            * | A1'*A1, A1'*A2, A1'*B |
            * | A2'*A1, A2'*A2, A2'*B |
            * | B'*A1,  B'*A2,   B'*B |
            *******************************************************************/
                // clang-format on
                H_.block(a1j, a1j, poseDim, poseDim) +=
                    A1.transpose() * A1 * w * rho[1];
                H_.block(a1j, a2j, poseDim, poseDim) +=
                    A1.transpose() * A2 * w * rho[1];
                H_.block(a1j, bj, poseDim, depthDim) +=
                    A1.transpose() * B * w * rho[1];

                H_.block(a2j, a1j, poseDim, poseDim) +=
                    A2.transpose() * A1 * w * rho[1];
                H_.block(a2j, a2j, poseDim, poseDim) +=
                    A2.transpose() * A2 * w * rho[1];
                H_.block(a2j, bj, poseDim, depthDim) +=
                    A2.transpose() * B * w * rho[1];

                H_.block(bj, a1j, depthDim, poseDim) +=
                    B.transpose() * A1 * w * rho[1];
                H_.block(bj, a2j, depthDim, poseDim) +=
                    B.transpose() * A2 * w * rho[1];
                H_.block(bj, bj, depthDim, depthDim) +=
                    B.transpose() * B * w * rho[1];
                // clang-format off
            /********************* 利用稀疏性计算g=-J'*b ****************************
            * | A1'|       | A1' * b |
            * | A2'| * b = | A2' * b |
            * | B' |       | B'  * b |
            **********************************************************************/
                // clang-format on
                g_.middleRows(a1j, poseDim) -= A1.transpose() * r * w * rho[1];
                g_.middleRows(a2j, poseDim) -= A2.transpose() * r * w * rho[1];
                g_.middleRows(bj, depthDim) -= B.transpose() * r * w * rho[1];
            }
        }
    }
    cout
        << "usefulNum, depthErrorNum, targetErrorNum, noInrangeNum, allBadNum: "
        << info.usefulNum << " " << depthErrorNum << " " << targetErrorNum
        << " " << noInrangeNum << " "
        << (depthErrorNum + targetErrorNum + noInrangeNum) << endl;
    info.cost /= info.usefulNum;
    return info;
}

double Optimizer::ConstructJ_H_b_g_byMatch() {
    // 构建H, g
    // 给出每个KF对应的在H矩阵中的位置
    // map<const KeyFrame*, int> kfMapCol;
    // map<const KeyFrame*, int> debugKFMapResidualNum;
    // for(int i = 0; i < window_.size(); ++i) {
    //     kfMapCol.insert({window_[i], i * 6});
    //     debugKFMapResidualNum.insert({window_[i], 0}); // 统计每个图像对应的residual数量
    // }
    // const int poseDim = window_[0]->Twc_.Size();
    // const int depthDim = 1;

    // // 或许我们不知道residual，Jacobian的行数，但是H矩阵以及g向量的维度是可知的
    // const int variableDim = window_.size() * poseDim + optLandmark_.size() * depthDim;
    // cout << "opt variable dim: " << variableDim << endl;
    // H_.resize(variableDim, variableDim);
    // H_.setZero();
    // g_.resize(variableDim);
    // g_.setZero();
    // double cost = 0;
    // int noInrangeNum = 0;
    // int depthErrorNum = 0;
    // int targetErrorNum = 0;
    // int usefulNum = 0;
    // // 计算residual & jacobian
    // /*********
    // *    T0 T1 ... d0 d1 ...
    // * r0
    // *********/
    // const int depthStartCol = window_.size() * poseDim;
    // constexpr int resDim = 1; // 添加了huberLoss
    // int resNum = 0; // 显示当前计算到雅可比的第几行
    // // TODO: 有很多landmark不能参与计算，需要将其排除在H及g信息之外，典型的如：
    // // opt variable dim: 22902
    // // usefulNum, depthErrorNum, targetErrorNum, noInrangeNum, allBadNum: 804 0 23938 1 23939
    // for(int i = 0; i < optLandmark_.size(); ++i) {
    //     Landmark *p = optLandmark_[i];
    //     KeyFrame *host = p->host_;
    //     const Eigen::Vector3d pc1 = p->GetPc();
    //     const Eigen::Vector3d pw = host->Twc_ * pc1;

    //     for(const auto &tar : p->target_) {
    //         // 不能向host投影，TODO: 删除host在target中的观测
    //         // if(tar.first == host || tar.first!=window_.back()) {
    //         if(tar.first == host) {
    //             ++targetErrorNum;
    //             continue;
    //         }
    //         KeyFrame *target = tar.first;

    //         const Eigen::Vector3d pc2 = target->Tcw_ * pw;
    //         if(pc2.z() < config->minDepth || pc2.z() > config->maxDepth) {
    //             ++depthErrorNum;
    //             continue;
    //         }
    //         const Eigen::Vector2d px2 = cam_->Project2PixelPlane(pc2);
    //         if(!InRange(target->dist_[0], px2.cast<int>()) ) {
    //             ++noInrangeNum;
    //             continue;
    //         }

    //         const Eigen::Vector2d r = px2 - p->target_[target];
    //         Eigen::Matrix<double, 1, 2> J_huber_r(0, 0);
    //         const double loss = HuberLoss(r, J_huber_r);

    //         ++usefulNum;
    //         cost += loss;
    //         debugKFMapResidualNum[target] += 1;
    //         resNum += resDim;

    //         /******** 投影过程 ********
    //         * K.inv * (u1, v1, 1) --> Pc1_norm * z1 --> Twc1 * Pc1 --> Twc2.inv * Pw -->
    //         * Pc2 / z2 -> K * Pc2_norm -> (u2, v2, 1) -> res(u2, v2)
    //         *
    //         * res w.r.t (u2, v2) [1x2]
    //         * (u2, v2) w.r.t Pc2_norm [2x3]
    //         * Pc2_norm w.r.t Pc2 [3x3]
    //         * Pc2 w.r.t Twc2 [3x6] ------> optimization variable
    //         * Pc2 w.r.t Pw [3x3]
    //         * Pw w.r.t Twc1 [3x6] -------> optimization variable
    //         * Pw w.r.t Pc1
    //         * Pc1 w.r.t z1 [3x1] --------> optimization variable
    //         ************************/

    //         // res w.r.t (u2, v2) [1x2]
    //         const Eigen::Matrix<double, 1, 2> J_res_px2 = J_huber_r;

    //         // px2 w.r.t Pc2 [2x3]
    //         const Eigen::Matrix<double, 2, 3> J_px2_Pc2Norm = p->cam_->K_.block(0, 0, 2, 3);
    //         Eigen::Matrix<double, 3, 3> J_Pc2Norm_Pc2;
    //         const double d = 1/pc2.z();
    //         const double d2 = 1./pow(pc2.z(), 2);
    //         J_Pc2Norm_Pc2 << d, 0, -pc2.x()*d2,
    //                         0, d, -pc2.y()*d2,
    //                         0, 0, 0;
    //         const Eigen::Matrix<double, 2, 3> J_px2_Pc2 = J_px2_Pc2Norm * J_Pc2Norm_Pc2;

    //         // FEJ
    //         if(!p->J_Pc2_Twc2.count(target)) {
    //         // if(!p->J_Pc2_Twc2.count(target) || 1) {
    //             // Pc2 w.r.t Twc2 : Pc2 = Twc2.inv * Pw
    //             Eigen::Matrix<double, 3, 6> J_Pc2_Twc2; // ------------------------> optimization variable
    //             const Eigen::Vector3d dt = pw - target->Twc_.t_wb_;
    //             // Pc2 w.r.t Rwc2
    //             J_Pc2_Twc2.block(0, 0, 3, 3) = SkewSymmetric(target->Tcw_.q_wb_ * dt); // Twc_.q_wb_.inverse()
    //             // Pc2 w.r.t Pwc2
    //             J_Pc2_Twc2.block(0, 3, 3, 3) = -target->Tcw_.q_wb_.toRotationMatrix(); // Twc.q_wb.R.transpose()

    //             // Pc2 w.r.t Pw
    //             // TODO: 这里也应该要使用首次的Tcw值吧！！！由于target有多帧，所以要保留多个
    //             const Eigen::Matrix3d J_Pc2_Pw = target->Tcw_.q_wb_.toRotationMatrix();

    //             if(p->J_Pc2_Pw.count(target)) {
    //                 // just for debug
    //                 p->J_Pc2_Twc2[target] = J_Pc2_Twc2;
    //                 p->J_Pc2_Pw[target] = J_Pc2_Pw;
    //             } else {
    //                 p->J_Pc2_Twc2.insert({target, J_Pc2_Twc2});
    //                 p->J_Pc2_Pw.insert({target, J_Pc2_Pw});
    //             }

    //             if(p->J_Pw_z.empty()) {
    //             // if(p->J_Pw_z.empty() || 1) {
    //                 // Pw w.r.t Twc1 : Pw = Twc1 * Pc1 = Rwc1 * pc1 + Pwc1
    //                 Eigen::Matrix<double, 3, 6> J_Pw_Twc1; // ------------------------> optimization variable
    //                 // Pw w.r.t Rwc1
    //                 J_Pw_Twc1.block(0, 0, 3, 3) = -host->Twc_.q_wb_.toRotationMatrix() * SkewSymmetric(pc1);
    //                 // Pw w.r.t Pwc1
    //                 J_Pw_Twc1.block(0, 3, 3, 3) = Eigen::Matrix3d::Identity();

    //                 // Pw w.r.t Pc1
    //                 const Eigen::Matrix3d J_Pw_Pc1 = host->Twc_.q_wb_.toRotationMatrix();

    //                 // Pc1 w.r.t z
    //                 const Eigen::Vector3d pc1Norm(p->GetPcNorm());
    //                 Eigen::Vector3d J_Pc1_z{pc1Norm.x(), pc1Norm.y(), 1}; // --------> optimization variable

    //                 if(!p->J_Pw_z.empty()) {
    //                     // just for debug
    //                     p->J_Pw_z.clear();
    //                     p->J_Pw_Twc1.clear();
    //                 }
    //                 p->J_Pw_z.push_back(J_Pw_Pc1 * J_Pc1_z);
    //                 p->J_Pw_Twc1.push_back(J_Pw_Twc1);
    //             }
    //         }

    //         // Residual w.r.t optimization variables Jacobian
    //         Eigen::Matrix<double, resDim, 6> A1 = J_res_px2 * J_px2_Pc2 * p->J_Pc2_Pw.at(target) * p->J_Pw_Twc1[0]; // J_res_Pw * J_Pw_Twc1;
    //         if(host == window_[0]) {
    //             // fixed滑动窗口第一帧
    //             //A1.setZero();
    //         }
    //         Eigen::Matrix<double, resDim, 6> A2 = J_res_px2 * J_px2_Pc2 * p->J_Pc2_Twc2.at(target); // J_res_Pc2 * J_Pc2_Twc2;
    //         if(target == window_[0]) {
    //             //A2.setZero();
    //         }
    //         const Eigen::Matrix<double, resDim, 1> B = J_res_px2 * J_px2_Pc2 * p->J_Pc2_Pw.at(target) * p->J_Pw_z[0]; // J_res_Pw * J_Pw_Pc1 * J_Pc1_z;
    //         const double w = 1.0; // /p->depthCov_;
    //         const int a1i = resNum, a1j = kfMapCol[host],
    //                   a2i = resNum, a2j = kfMapCol[target],
    //                   bi = resNum, bj = depthStartCol + i;
    //         /********************* 利用稀疏性计算H=J'*J ****************************
    //         * | A1'|
    //         * | A2'| * | A1 A2 B |
    //         * | B' |
    //         * =
    //         * | A1'*A1, A1'*A2, A1'*B |
    //         * | A2'*A1, A2'*A2, A2'*B |
    //         * | B'*A1,  B'*A2,   B'*B |
    //         *******************************************************************/
    //         H_.block(a1j, a1j, poseDim, poseDim) += A1.transpose() * A1 * w;
    //         H_.block(a1j, a2j, poseDim, poseDim) += A1.transpose() * A2 * w;
    //         H_.block(a1j, bj, poseDim, depthDim) += A1.transpose() * B * w;

    //         H_.block(a2j, a1j, poseDim, poseDim) += A2.transpose() * A1 * w;
    //         H_.block(a2j, a2j, poseDim, poseDim) += A2.transpose() * A2 * w;
    //         H_.block(a2j, bj, poseDim, depthDim) += A2.transpose() * B * w;

    //         H_.block(bj, a1j, depthDim, poseDim) += B.transpose() * A1 * w;
    //         H_.block(bj, a2j, depthDim, poseDim) += B.transpose() * A2 * w;
    //         H_.block(bj, bj, depthDim, depthDim) += B.transpose() * B * w;
    //         /********************* 利用稀疏性计算g=-J'*b ****************************
    //         * | A1'|       | A1' * b |
    //         * | A2'| * b = | A2' * b |
    //         * | B' |       | B'  * b |
    //         **********************************************************************/
    //         g_.middleRows(a1j, poseDim) -= A1.transpose() * loss * w;
    //         g_.middleRows(a2j, poseDim) -= A2.transpose() * loss * w;
    //         g_.middleRows(bj, depthDim) -= B.transpose() * loss * w;

    //     }
    // }

    // if(config->relativePoseConstraintWeight > 0.) {
    //     Eigen::MatrixXd H;
    //     Eigen::VectorXd g;
    //     ConstructRelativePoseConstraint(H, g);
    //     H_.block(0, 0, H.rows(), H.cols()) += H;
    //     g_.middleRows(0, g.rows()) += g;
    // }
    // cout << "usefulNum, depthErrorNum, targetErrorNum, noInrangeNum, allBadNum: " << usefulNum << " "
    //      << depthErrorNum << " " << targetErrorNum << " " << noInrangeNum << " "
    //      << (depthErrorNum + targetErrorNum + noInrangeNum) << endl;
    double cost = 0;
    return cost;
}

void Optimizer::ConstructRelativePoseConstraint(Eigen::MatrixXd& H,
                                                Eigen::VectorXd& g) {
    if (config->relativePoseConstraintWeight <= 0.) {
        return;
    }
    const int PoseDim = window_[0]->Twc_.Size();
    const int resDim = 6;

    H.resize(window_.size() * PoseDim, window_.size() * PoseDim);
    g.resize(window_.size() * PoseDim);
    H.setZero();
    g.setZero();
    const double w = config->relativePoseConstraintWeight;
    for (int i = 1; i < window_.size(); ++i) {
        KeyFrame* kf1 = window_[i - 1];
        KeyFrame* kf2 = window_[i];
        const Pose pTwc1 = kf1->priorTwc_;
        const Pose pTwc2 = kf2->priorTwc_;
        const Pose pTc1c2 = pTwc1.Inverse() * pTwc2;
        const Pose Tc1c2 = kf1->Twc_.Inverse() * kf2->Twc_;
        const Eigen::Quaterniond deltaQ(pTc1c2.q_wb_.inverse() * Tc1c2.q_wb_);
        Eigen::Matrix<double, 6, 1> r_R_P;
        // 实际是 ΔR * Rwc1.inv * Rwc2
        r_R_P.head(3) = LogSO3(deltaQ.toRotationMatrix());
        // 实际是 ΔP - Rwc1.T*Pwc1 + Rwc1.T*Pwc2
        r_R_P.tail(3) = pTc1c2.t_wb_ - Tc1c2.t_wb_;

        const int aj1 = (i - 1) * PoseDim, aj2 = i * PoseDim;
        Eigen::Matrix<double, 6, 6> A1, A2;
        A1.setZero();
        A2.setZero();

        // 使用"BCH近似"之前，需要通过"伴随性质"将扰动量换到右边
        // dLogSO3(ΔR*R1.T*R2) ---> 微分扰动
        // = LogSO3(ΔR*exp(-ε1^)*R1.T*R2) ---> 使用伴随:
        // = LogSO3(ΔR*R1.T*R2 * Exp(-R2.T*R1*ε1)) ---> Exp{小量}，使用BCH近似
        // = [Jr(ΔR*R1.T*R2).inv * -R2.T*R1*ε1] + LogSo3(ΔR*R1.T*R2)
        const Eigen::Matrix3d invJr = InverseRightJacobianSO3(r_R_P.head(3));
        const Eigen::Vector3d dt = kf1->Twc_.t_wb_ - kf2->Twc_.t_wb_;
        const Eigen::Matrix3d R1 = kf1->Twc_.q_wb_.toRotationMatrix();
        const Eigen::Matrix3d R2 = kf2->Twc_.q_wb_.toRotationMatrix();

        // LogSO3(ΔR * Rwc1.inv * Rwc2) w.r.t Rwc1
        A1.block(0, 0, 3, 3) = -invJr * R2.transpose() * R1;

        // LogSO3(ΔR * Rwc1.inv * Rwc2) w.r.t Rwc2
        A2.block(0, 0, 3, 3) = invJr;

        // ΔP + Rwc1.T*(Pwc1 - Pwc2) w.r.t Rwc1
        A1.block(3, 0, 3, 3) = SkewSymmetric(R1.transpose() * dt);

        // ΔP + Rwc1.T*(Pwc1 - Pwc2) w.r.t Rwc2

        // ΔP + Rwc1.T*(Pwc1 - Pwc2) w.r.t Pwc1
        A1.block(3, 3, 3, 3) = R1.transpose();

        // ΔP + Rwc1.T*(Pwc1 -Pwc2) w.r.t Pwc2
        A2.block(3, 3, 3, 3) = -R1.transpose();

        H.block(aj1, aj1, 6, 6) += A1.transpose() * A1 * w;
        H.block(aj1, aj2, 6, 6) += A1.transpose() * A2 * w;
        H.block(aj2, aj2, 6, 6) += A2.transpose() * A2 * w;
        H.block(aj2, aj1, 6, 6) += A2.transpose() * A1 * w;
        g.middleRows(aj1, 6) -= A1.transpose() * r_R_P * w;
        g.middleRows(aj2, 6) -= A2.transpose() * r_R_P * w;
    }
}

bool Optimizer::SlidingWindowOptimize() {
    cout << "Begin SlidingWindowOptimize!!!" << endl;
    if (!SetOptimizeVariables()) {
        cerr << "find landmark for optimization num: " << optLandmark_.size()
             << " too small " << endl;
        // return false;
    }
    int margKFid = SelectOneKF2Marginalization();
    cout << "margKFid: " << margKFid << endl;

    const int sampleNum = SampleUsefulLandmark();
    cout << "Sample landmark num: " << sampleNum << endl;
    return ExecuteWindowOptimize();
}

void Optimizer::HuberLoss(const double chi2, Eigen::Vector2d& rho,
                          const int lvl) {

    //const double scale = 1.0 / pow(2, lvl);
    const double huberDelta = config->huberDelta;  //  * scale;

    const double huberDelta2 = huberDelta * huberDelta;
    if (chi2 > huberDelta2) {
        // r = δ*(|a|-0.5*δ), 这里的loss形式为：
        // loss = hb*|f(x) - obv| - 0.5*hb^2{写成向量乘法为0.5*hb.T*hb}, f(x)非线性，在x0处展开有：
        // loss = hb*|f(x0) + J*Δx -obv| - 0.5*hb^2，记: r = f(x0) - obv，则：
        // loss = hb*|J*Δx + r| - 0.5*hb*hb{点积形式}, 目标是让 loss = 0，那么，当|J*Δx + r| >= 0时，
        // 根据矩阵乘法及点积运算规则，有：
        // loss = hb.T*J*Δx + hb.T*r - 0.5*hb.T*hb, 欲让loss最小值为0，则：
        // hb.T*J*Δx = -hb.T*r + 0.5*hb.T*hb
        // 等式两边同乘以J.T，有：
        // hb.T* J.T*J*Δx = -J.T * (hb.T*r - 0.5*hb.T*hb)
        // 当hb是一维的时候，有：
        // J.T*J*Δx = -J.T*(r - 0.5*hb)
        // 同理，当 |J*Δx + r| < 0时，
        // loss = -hb.T*J*Δx - hb.T*r - 0.5hb.T*hb，欲让loss为，则：
        // hb.T*J*Δx = -hb.T*r - 0.5*hb.T*hb
        // 两边同乘以J.T，并约去hb：
        // J.T*J*Δx = -J.T*(r + 0.5*hb)
        // 提问：当hb是2维及以上向量的时候，如何处理？
        // 答：求向量的逆，即有 hb.inv * hb.T = Identity，由于向量没有逆矩阵，这种方法不可行
        // 另一种想法是：由于 a.dot(b) = |a|*|b|*cosθ，当 a平行于b时取得最大值，若|r|>|hb|，
        // 那么， r.dot(hb) = |a|*|hb|*cosθ<a, hb>，这时可能出现负值，使得残差下降，这种方式是不行的，因为最小化目的是让残差为0，
        // 所以我原本计算残差值的想法才是对的？？？

        // OK，对于向量形式，以i2维向量为例，我们定义hb=(h1, h2).T{h1, h2 > 0}，残差为：
        // hb.T * |r| - 0.5 hb.T*hb，我们只需要对残差r各个维度取绝对值即可就记为|r|，那么可以证明：
        // hb与|r|同属于第一象限，夹角小于90度，即hb.T.dot(|r|) < r.dot(r)且均>0
        // 存在的问题是： |r|*cos<r, hb> < 0.5*|hb|，那么这个时候，loss = hb.T * |r| - 0.5*hb.T*hb还是会出现负值
        // 或者直接定义： loss = 0.5*hb.T*|r|这样可以保证其小于 0.5*r.T*r
        // 即 loss = 0.5 * hb.T * |r| = 0.5 * hb.T * |f(x0) + JΔx - obv|，这个|r|还是无法展开

        // 如果每个维度都单独考虑呢？ 那么有： loss = hb.T*|f(x) - obv| - 0.5*hb.T*hb
        // loss = hb.T * |JΔx + r| -0.5

        // 经过查看 g2o源码，发现我原来的理解才是正确的，胡伯核函数只能是关于标量的实现
        // 因为这里胡伯核函数值关于状态量的关系简单，不需要像投影函数那样再线性化
        rho[0] = sqrt(chi2) * huberDelta - 0.5 * huberDelta2;
        rho[1] = huberDelta / sqrt(chi2);
    } else {
        // r = 0.5*a^2 // 这里描述的是最终的残差形式
        // 这里的loss是一般形式，即 loss = 0.5 * (f(x) - obv)^2，f(x)是非线性的，在x0处展开，设f(x0)'=J，有：
        // loss = 0.5 * (f(x0) + JΔx - obv)^2, 记 r = f(x0)-obv，那么：
        // loss = 0.5 * (JΔx + r)^2 = 0.5 * (Δx.T*J.T*J*Δx + 2*Δx.T*J.T*r + r*r)，易知极小值点在dloss/dΔx = 0处，则有：
        // 2*J.T*J*Δx + 2*J.T*r = 0 ==>
        // J.T*J*Δx = -J.T*r
        rho[0] = chi2;
        rho[1] = 1;
    }
}

int Optimizer::SelectOneKF2Marginalization() {
    // Keep the last 2 frame
    if (static_cast<int>(window_.size()) <= config->maxKFnumInWindow) {
        return 1000;
    }
    // 不考虑结构的情况下，移除掉与最新帧观测最少的
    // TODO: 有多帧小于可删除阈值时，考虑删除关键点分布较差的KF
    const int keepLastKFnum = config->keepLastKFnumInWindow;
    vector<double> score(window_.size() - keepLastKFnum, 0);
    for (int i = 0; i < static_cast<int>(window_.size()) - keepLastKFnum; ++i) {
        vector<Landmark*>& ps = window_[i]->landmark_;
        double convergeNum = 0;
        double seenByNewestNum = 0;
        for (Landmark* p : ps) {
            if (p == nullptr || !p->Converge() || p->IsOutOfRange()) {
                continue;
            }
            convergeNum += 1;

            const Eigen::Vector3d pc2 = window_.back()->Tcw_ * p->GetPw();
            const Eigen::Vector2d px2 = cam_->Project2PixelPlane(pc2);
        }
        score[i] = seenByNewestNum / convergeNum;
    }
    double smallRatio = DBL_MAX;
    int smallId = 0;
    cout << "score: ";
    for (size_t i = 0; i < score.size(); ++i) {
        cout << score[i] << " ";
        if (score[i] < smallRatio) {
            smallId = i;
            smallRatio = score[i];
        }
    }
    cout << endl;
    KeyFrame* oldest = window_[smallId];
    window_[smallId] = window_[0];
    window_[0] = oldest;
    return smallId;
}

bool Optimizer::TrackLocalMap(KeyFrame* kf2, bool& needNewKFbySight) {

    const Pose noise = ConvertRPYandPostion2Pose({0.01, 0.02, 0}, {0.01, 0.02, 0}, kDeg2Rad);
    Pose Twc2 = kf2->Twc_ * noise;
    // 仅优化当前帧pose，避免由于其运动模糊影响landmark估计值导致系统崩溃
    // 同时加快计算速度
    onlyPoseUpdate_ = true;
    OptimizeCurFrame(window_.back()->optFlw_, Twc2, kf2->id_);
    onlyPoseUpdate_ = false;

    Pose beforeTwc2 = kf2->Twc_;
    kf2->SetTwc(Twc2);
    cout << "cur frame pose diff: " << beforeTwc2.Inverse() * kf2->Twc_ << endl;
    return true;
}

void Optimizer::CullingErrorLandmark(KeyFrame* curF) {
    // TODO：如何剔除异常点
    /*
    int winSize = window_.size();  // 为避免size_t对应的负数是极大正数
    if (curF == nullptr) {
        // pose正确的前提下，如果是合理的深度，那么投影到当前帧的地图点必须要正常
        // 首先按照id排序避免距离过远的KF进行重复投影
        sort(window_.begin(), window_.end(),
             [](KeyFrame* a, KeyFrame* b) { return a->id_ < b->id_; });

        // 每个KF检查depth两次
        const int checkNum = config->maxKFnumInWindow - 2;  // 2;
        for (int i = winSize - 1 - checkNum; i < winSize - 1 && i >= 0; ++i) {
            const double badDepthRatio =
                window_[i]->CullingBadDepth(window_.back());
            cout << i << " KF badDepthRatio vs newestKF: " << badDepthRatio
                 << endl;
        }
    } else {
        for (int i = 0; i < winSize; ++i) {
            if (i < 0)
                continue;
            const double badDepthRatio = window_[i]->CullingBadDepth(curF);
            cout << i << " KF badDepthRatio vs curF: " << badDepthRatio << endl;
        }
    }
    */
}

void Optimizer::RemoveOneKeyframe(const KeyFrame& curF) {
    // 1. 如果当前帧与上上一帧有足够的水平距离，就移除最老帧
    // 2. 否则移除最近帧以保证视差
    if (static_cast<int>(window_.size()) < config->maxKFnumInWindow) {
        return;
    }

    // 说明预设关键帧数量较少
    if (window_.size() < 3) {
        window_.erase(window_.begin());
        return;
    }

    const Eigen::Vector3d posDiff =
        curF.Twc_.t_wb_ - window_[window_.size() - 2]->Twc_.t_wb_;
    const double horDist = posDiff.head(2).norm();
    cout << "curF hor dist to the last 2 frame: " << horDist << endl;
    if (horDist < config->needNewKFtrans * 1.5) {
        window_.pop_back();
        cout << "Remove the last keyframe from window!";
        return;
    }

    window_.erase(window_.begin());
    cout << "Remove the first keyframe from window!";
}

void Optimizer::DrawTriangulateCase(const double estD1, const Landmark& lk1,
                                    const Eigen::Vector2i& matchKp2,
                                    const cv::Mat& debugImg2, const Pose& T12,
                                    const bool success) {
    const KeyFrame* host = lk1.host_;
    const cv::Mat& debugGrayImg_ = host->debugGrayImg_;
    cv::Mat showImg(debugGrayImg_.rows, debugGrayImg_.cols * 2, CV_8UC3,
                    cv::Scalar{0, 0, 0});
    cv::Mat im1, im2;
    cvtColor(debugGrayImg_, im1, cv::COLOR_GRAY2BGR);
    cvtColor(debugImg2, im2, cv::COLOR_GRAY2BGR);
    im1.copyTo(showImg.colRange(0, debugGrayImg_.cols));
    im2.copyTo(showImg.colRange(debugGrayImg_.cols, showImg.cols));

    int start_text_row = 20;
    int step_text_row = 20;
    cv::putText(showImg, T12.QwbString(), cv::Point(10, (start_text_row)),
                cv::FONT_ITALIC, 0.8, kColor.at("red"), 1);
    cv::putText(showImg, T12.PwbString(),
                cv::Point(10, (start_text_row += step_text_row)),
                cv::FONT_ITALIC, 0.8, kColor.at("red"), 1);
    const string caseName = success ? "Suc tri" : "Fai tri";
    cv::putText(showImg,
                fmt::format("{}_kf_id:{}", caseName, to_string(lk1.host_->id_)),
                cv::Point(10, (start_text_row += step_text_row)),
                cv::FONT_ITALIC, 0.8, kColor.at("red"), 1);

    const cv::Vec3b& matchColor = kColor.at("yellow");

    int radius = 3;

    cv::Vec3b nearColor(0, 255, 0);
    cv::Vec3b farColor(0, 0, 255);
    const cv::Point pointDiff(debugGrayImg_.cols, 0);
    const Eigen::Vector2i& kp1 = lk1.uv_.cast<int>();
    cv::Point p1(kp1.x(), kp1.y());
    cv::Point p2(matchKp2.x(), matchKp2.y());
    constexpr double kTextRatio = 0.5;
    const cv::Point textDiff(5, 0);
    // 写必要信息
    cv::putText(showImg,
                fmt::format("({}, {}, {:.1f}, {:.1f}, {})", p1.x, p1.y,
                            1.0 / estD1, lk1.trueDepth_, lk1.obvTime_ + 1),
                p1 + textDiff, cv::FONT_ITALIC, kTextRatio, kColor.at("red"),
                1);

    // 画极线起终点，起点绿色，终点红色，连线蓝色
    cv::line(showImg, p1, p2 + pointDiff, matchColor, 1);

    // 画极线以查看匹配是否准确
    cv::circle(showImg, p1, radius, matchColor, 1);
    cv::circle(showImg, p2 + pointDiff, radius, matchColor, 1);

    const string debugImgName = fmt::format("{}_{}", lk1.uv_.x(), lk1.uv_.y());
    triPointMapDebugImage_[debugImgName].emplace_back(showImg);
}

void Optimizer::DrawProjectCase(const Landmark& lk1,
                                const Eigen::Vector2d& matchKp2,
                                const cv::Mat& debugImg2, const Pose& Twc2) {
    const KeyFrame* host = lk1.host_;
    const cv::Mat& debugGrayImg_ = host->debugGrayImg_;
    cv::Mat showImg(debugGrayImg_.rows, debugGrayImg_.cols * 2, CV_8UC3,
                    cv::Scalar{0, 0, 0});
    cv::Mat im1, im2;
    cvtColor(debugGrayImg_, im1, cv::COLOR_GRAY2BGR);
    cvtColor(debugImg2, im2, cv::COLOR_GRAY2BGR);
    im1.copyTo(showImg.colRange(0, debugGrayImg_.cols));
    im2.copyTo(showImg.colRange(debugGrayImg_.cols, showImg.cols));

    int start_text_row = 20;
    int step_text_row = 20;
    const Pose T12 = host->Tcw_ * Twc2;
    cv::putText(showImg,
                fmt::format("kf id: {}, {}", host->id_, T12.PwbString()),
                cv::Point(10, (start_text_row)), cv::FONT_ITALIC, 0.8,
                kColor.at("red"), 1);
    cv::putText(showImg, T12.QwbString(),
                cv::Point(10, (start_text_row += step_text_row)),
                cv::FONT_ITALIC, 0.8, kColor.at("red"), 1);

    const cv::Point pointDiff(debugGrayImg_.cols, 0);
    const cv::Point textDiff(5, 0);
    const Eigen::Vector3d pc2 = Twc2.Inverse() * lk1.GetPw();
    const Eigen::Vector2d px = cam_->Project2PixelPlane(pc2);
    const double residual = (px - matchKp2).norm();
    cv::Point pxi(int(px.x()), int(px.y()));
    cv::Point kp2i(int(matchKp2.x()), int(matchKp2.y()));
    cv::putText(showImg, fmt::format("r: {:.1f}", residual),
                pointDiff + pxi + textDiff, cv::FONT_ITALIC, 0.8,
                kColor.at("red"), 1);

    const cv::Vec3b& matchColor = kColor.at("yellow");

    int radius = 3;

    const Eigen::Vector2d& kp1 = lk1.uv_;
    cv::Point p1i(int(kp1.x()), int(kp1.y()));
    constexpr double kTextRatio = 0.5;
    // 写必要信息
    cv::putText(showImg,
                fmt::format("({}, {}, {:.1f}, {:.1f}, {})", p1i.x, p1i.y,
                            1.0 / lk1.invZ_, lk1.trueDepth_, lk1.obvTime_ + 1),
                p1i + textDiff, cv::FONT_ITALIC, kTextRatio, kColor.at("red"),
                1);

    // 画极线起终点，起点绿色，终点红色，连线蓝色
    cv::line(showImg, p1i, kp2i + pointDiff, matchColor, 1);

    // 画极线以查看匹配是否准确
    cv::circle(showImg, p1i, radius, matchColor, 1);
    cv::circle(showImg, kp2i + pointDiff, radius, matchColor, 1);
    cv::circle(showImg, pxi + pointDiff, radius, kColor.at("red"), 1);

    const string debugImgName = fmt::format("{}_{}", lk1.uv_.x(), lk1.uv_.y());
    triPointMapDebugImage_[debugImgName].emplace_back(showImg);
}

void Optimizer::ShowLocalMap() {
    if (window_.size() < 1) {
        return;
    }

    vector<Pose> vTwc;
    for (int i = 0; i < static_cast<int>(window_.size()); ++i) {
        KeyFrame* kf = window_[i];
        vTwc.push_back(kf->Twc_);
    }

    set<Landmark*>& aPoints = interaction->activePoints;
    set<Landmark*>& lPoints = interaction->localPoints;
    aPoints.clear();
    lPoints.clear();
    for (Landmark* p : optLandmark_) {
        if (p != nullptr && !aPoints.count(p) && !p->IsOutOfRange() &&
            p->ManySupport() && p->Converge()) {
            aPoints.insert(p);
        }
    }

    for (int i = 0; i < window_.size(); ++i) {
        // for(int i = window_.size()-1; i < window_.size(); ++i) {
        KeyFrame* kf = window_[i];
        for (Landmark* p : kf->landmark_) {
            if (p != nullptr && !aPoints.count(p) && !lPoints.count(p) &&
                !p->IsOutOfRange() && p->ManySupport() && p->Converge()) {
                lPoints.insert(p);
            }
        }
    }

    if (!aPoints.empty() || !lPoints.empty()) {
        ::ShowLocalMap(vTwc);
        // cout << "show " << vTwc.size() << " KFs " << (aPoints.size()+lPoints.size()) << " map points" << endl;
    } else {
        //cerr << "wait for local map..." << endl;
    }
}
