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

constexpr int kMinUsefulObvNum = 2;  // 扣除host的观测
constexpr int kMinUsefulObvNumWithHost = kMinUsefulObvNum + 1;

Optimizer::Optimizer(shared_ptr<Camera> cam, const double lambda,
                     const int maxIte, const bool onlyPoseUpdate)
    : lambda_(lambda),
      maxIte_(maxIte),
      onlyPoseUpdate_(onlyPoseUpdate),
      cam_(cam) {
    maxIte_ = config->maxIterationLM;
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
    const std::vector<Eigen::Vector2d>& obvs, const Pose& Twc2) {

    ResidualInfo info;
    string debugInfo("chi2 residuals: ");
    constexpr int kStepInfoOut = 300000;

    const Camera& cam = *cam_;
    const Pose Tc2w = Twc2.Inverse();
    for (size_t j = 0; j < lk1s.size(); ++j) {
        Landmark* lk1 = lk1s[j];
        if (lk1->NoUsed()) {
            continue;
        }

        const Eigen::Vector3d pc = Tc2w * lk1->GetPw();
        const Eigen::Vector2d px = cam.Project2PixelPlane(pc);

        // 必须与计算Jacobian的残差计算方式一致
        // 注意： cost = p.T * p = [1x1]向量，我们是对cost进行线性化，因此求导的对象是r^2，
        // 而胡伯核函数的自变量是r^2
        double chi2 = (px - obvs[j]).squaredNorm();
        Eigen::Vector2d rho;  // 残差值和核函数关于残差的导数
        HuberLoss(chi2, rho);
        info.cost += rho[0];
        ++info.totalConstraintNum;
        ++info.usefulLandmarkNum;  // 这里每个地图点只会投影一次到当前帧
        if (j % kStepInfoOut == 0) {
            debugInfo.append(fmt::format(
                "chi2: {:.1f}-rho[0]: {:.1f}-kf id: {}-kp1:({:.0f}, "
                "{:.0f}); ",
                chi2, rho[0], lk1->host_->id_, lk1->uv_.x(), lk1->uv_.y()));
        }
    }

    info.meanCost = info.cost / info.totalConstraintNum;
    //chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    //const double spendTime = chrono::duration<double>(t1 - t0).count();

    //cout << fmt::format(
    //    "curF BA: residual info.all.cost: {:.1f}, useful landmark num: {}, "
    //    "total constraint num: {}, mean cost: {:.1f}, spend time: {}\ndebug "
    //    "residual info: {}\n",
    //    info.cost, info.usefulLandmarkNum, info.totalConstraintNum, meanCost,
    //    spendTime, debugInfo);
    if (isnan(info.cost) || isinf(info.cost)) {
        info.cost = DBL_MAX;
    }
    return info;
}

Optimizer::ResidualInfo Optimizer::SetOptimizeLandmarkForTracking(
    const std::vector<Landmark*>& lk1s,
    const std::vector<Eigen::Vector2d>& obvs, const Pose& Twc2,
    const cv::Mat& img, int& canUseNum, std::vector<Landmark*>& stableLks,
    std::vector<Eigen::Vector2d>& stableObvs) {

    ResidualInfo info;
    string debugInfo("chi2 residuals: ");
    constexpr int kStepInfoOut = 300000;

    // 根据观测数分，>1, >2, >3
    vector<vector<pair<int, double>>> resampleStableLkIndex2Chi2(3);
    chrono::steady_clock::time_point t0 = chrono::steady_clock::now();

    for (auto& vec : resampleStableLkIndex2Chi2) {
        vec.reserve(lk1s.size());
    }

    const Camera& cam = *cam_;
    const Pose Tc2w = Twc2.Inverse();
    const double maxChi2 = config->maxProjectError * config->maxProjectError;
    for (size_t j = 0; j < lk1s.size(); ++j) {
        Landmark* lk1 = lk1s[j];
        if (lk1->NoUsed()) {
            continue;
        }

        const Eigen::Vector3d pc = Tc2w * lk1->GetPw();
        const Eigen::Vector2d px = cam.Project2PixelPlane(pc);
        const bool inRange = InRange(img, px.cast<int>());
        if (inRange && pc.z() > kMinSceneDepthInCamera) {
            // 必须与计算Jacobian的残差计算方式一致
            // 注意： cost = p.T * p = [1x1]向量，我们是对cost进行线性化，因此求导的对象是r^2，
            // 而胡伯核函数的自变量是r^2
            double chi2 = (px - obvs[j]).squaredNorm();
            if (chi2 > maxChi2) {
                lk1->SetNoUsed();
                continue;
            }

            Eigen::Vector2d rho;  // 残差值和核函数关于残差的导数
            HuberLoss(chi2, rho);
            if (j % kStepInfoOut == 0) {
                debugInfo.append(fmt::format(
                    "curF set opt landmark chi2: {:.1f}-rho[0]: {:.1f}-kf id: "
                    "{}-kp1:({:.0f}, "
                    "{:.0f}); ",
                    chi2, rho[0], lk1->host_->id_, lk1->uv_.x(), lk1->uv_.y()));
            }

            const int id = lk1->target_.size();
            switch (id) {
                case 2:
                    resampleStableLkIndex2Chi2[0].emplace_back(j, rho[0]);
                    break;
                case 3:
                    resampleStableLkIndex2Chi2[1].emplace_back(j, rho[0]);
                    break;
                default:
                    resampleStableLkIndex2Chi2[2].emplace_back(j, rho[0]);
                    break;
            }

        } else {
            lk1->SetNoUsed();
        }
    }

    // 由观测数量由高到低进行采样
    constexpr int kMaxSampleLandmarkNum = 20000;
    canUseNum = 0;
    for (const auto& vec : resampleStableLkIndex2Chi2) {
        canUseNum += vec.size();
    }
    for (int i = 2; i >= 0; --i) {
        for (const pair<int, double>& idx2Chi2 :
             resampleStableLkIndex2Chi2[i]) {

            stableLks.emplace_back(lk1s[idx2Chi2.first]);
            stableObvs.emplace_back(obvs[idx2Chi2.first]);
            info.cost += idx2Chi2.second;
            ++info.totalConstraintNum;
            ++info.usefulLandmarkNum;  // 这里每个地图点只会投影一次到当前帧
            if (info.totalConstraintNum > kMaxSampleLandmarkNum) {
                break;
            }
        }

        if (info.totalConstraintNum > kMaxSampleLandmarkNum) {
            break;
        }
    }

    info.meanCost = info.cost / info.totalConstraintNum;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    const double spendTime = chrono::duration<double>(t1 - t0).count();

    cout << fmt::format(
        "curF BA: set opt tracking variable residual info.all.cost: {:.1f}, "
        "useful landmark num: {}, "
        "total constraint num: {}, mean cost: {:.1f}, spend time: {}\ndebug "
        "residual info: {}\n",
        info.cost, info.usefulLandmarkNum, info.totalConstraintNum,
        info.meanCost, spendTime, debugInfo);
    if (isnan(info.cost) || isinf(info.cost)) {
        info.cost = DBL_MAX;
    }
    return info;
}

void Optimizer::CalculateHandGradiantCurFrame(
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
    J_px2_Pc2Norm(0, 2) = 0.;
    J_px2_Pc2Norm(1, 2) = 0.;

    for (size_t j = 0; j < lk1s.size(); ++j) {
        Landmark& p = *lk1s[j];
        if (p.NoUsed()) {
            continue;
        }
        const Eigen::Vector3d Pw1 = p.GetPw();
        const Eigen::Vector3d Pc2 = Tc2w * Pw1;
        const Eigen::Vector2d px2 = p.cam_->Project2PixelPlane(Pc2);

        const Eigen::Vector2d r = px2 - obvs[j];
        double chi2 = r.squaredNorm();
        Eigen::Vector2d rho;
        HuberLoss(chi2, rho);

        Eigen::Matrix<double, 3, 3> J_Pc2Norm_Pc2;
        const double d = 1 / Pc2.z();
        const double d2 = 1. / pow(Pc2.z(), 2);
        J_Pc2Norm_Pc2 << d, 0, -Pc2.x() * d2, 0, d, -Pc2.y() * d2, 0, 0, 0;
        Eigen::Matrix<double, 2, 3> J_px2_Pc2 = J_px2_Pc2Norm * J_Pc2Norm_Pc2;

        // Pc2 w.r.t T12 [3x6]
        Eigen::Matrix<double, 3, 6> J_Pc2_Twc2 =
            Eigen::Matrix<double, 3, 6>::Zero();

        const Eigen::Vector3d dt = Pw1 - Twc2.t_wb_;
        // * Pc2 w.r.t R12
        J_Pc2_Twc2.block(0, 0, 3, 3) = SkewSymmetric(Tc2w.q_wb_ * dt);
        // * Pc2 w.r.t t12
        J_Pc2_Twc2.block(0, 3, 3, 3) = -Tc2w.q_wb_.toRotationMatrix();

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
}

Eigen::VectorXd Optimizer::SchurCompleteSolve(
    const Eigen::MatrixXd& H, const Eigen::VectorXd& b, const int poseNum,
    const int pointNum, const int poseDim, const int pointDim,
    const bool& logOut) {
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
        if (abs(D(i, i)) > 1e-9) {
            Dinv(i, i) = 1.0 / D(i, i);
        }
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
    if (logOut) {
        cout << "A:\n"
             << A << endl
             << "B:\n"
             << B << endl
             << "C:" << C.diagonal().transpose() << endl
             << "D:" << D.diagonal().transpose() << endl
             << "Dinv:" << Dinv.diagonal().transpose() << endl
             << "E:\n"
             << E << endl
             << "newA:\n"
             << newA << endl
             << "newb: " << new_b.transpose() << endl;
    }
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

    // cout << setprecision(5) << "deltaPose: "
    //      << deltaPose.head(poseNum * window_[0]->Tcw_.Size()).transpose()
    //      << "\n";
    // cout << setprecision(5) << "deltaPoint: " << deltaPoint.head(10).transpose()
    //      << "\n";

    // fmt::format(
    //     "calculate D.inv spend: {:.1f}, calculate E mat spend: {:.1f}, "
    //     "calculate left mat spend:{:.1f}, calculate dPose spend: {:.1f}, "
    //     "calculate dPoint spend: {:.1f}, construct dX spend: {:.1f}\n",
    //     ChronoTimeDurationCal(t0, t1), ChronoTimeDurationCal(t1, t1_1),
    //     ChronoTimeDurationCal(t1_1, t2), ChronoTimeDurationCal(t2, t3),
    //     ChronoTimeDurationCal(t3, t4), ChronoTimeDurationCal(t4, t5));
    return deltaX;
}

double Optimizer::CalculatePriorCost(const Eigen::VectorXd& deltaX) {
    // 边缘化与执行优化时的状态量数量要一致，
    // 因此需要先确定optLandmark的数量，难点在于如何处理转移成功且观测数大于2的优化点？
    // 可以先转移所有权，再在构建完先验约束后，删除对于被边缘化帧的观测，此时，只在边缘化构建先验信息时，设置Landmark*是否要为noUsed
    // 公式推导：
    // H_p * deltaX0 = g_p，==>  H_p * deltaX0 - g_p = 0
    // J_p.T * J_p * deltaX0 = -J_p.T * r_p
    // 因此: J_p * deltaX0 = -r_p
    // 所以，构建先验信息时刻，有 J_p * deltaX0 + r_p = 0
    // 构建先验残差项： (需要0.5系数才能与重投影残差权重一致)
    // C_p = 0.5 * (J_p * deltaX + r_p - 0).T * (J_p * deltaX + r_p - 0)
    // C_p = 0.5 * deltaX.T * J_p.T * J_p * deltaX + deltaX.T * J_p.T * r_p + 0.5 * r_p.T * r_p
    // C_p = 0.5 * deltaX.T * H_p * deltaX - deltaX.T * g_p + 常数
    // 注意：当线性化点一直在变时，需要记住构建先验残差时的状态量：
    // 由于： deltaX1 = X1 - X0
    // deltaX2 = X2 - X1
    // ...依次类推，最终：
    // deltaXn +...+ deltaX2 + deltaX1 = Xn - X0
    // 所以每次让deltaX0 += deltaX 进行更新即可，需要注意，当最终更新不被接受时，需要重置回上一次的deltaX0
    const Eigen::VectorXd deltaNew = margDeltaX_ + deltaX;
    //return 0.5 * deltaNew.transpose() * (Hp_ * deltaNew - g_p_) + 0.5 * rpChi2_;
    // 先验残差为负是可允许的，常数项0.5*rp.T*rp不影响整体优化方向，不需要考虑
    return 0.5 * deltaNew.transpose() * (Hp_ * deltaNew - g_p_);
}

void Optimizer::UpdatePriorDeltaX0(const Eigen::VectorXd& deltaX) {
    margDeltaX_ += deltaX;
}

bool Optimizer::ExecuteWindowOptimize() {
    ResidualInfo lastCost;
    if (margKFstatus_) {
        // 边缘化结束时，已设置好滑窗优化变量
        lastCost = CalculateResidualWindow(false);
    } else {
        lastCost = SetOptimizeStatusVariableForWindowBA();
    }

    // 丢失追踪，重新进行
    if (lastCost.usefulLandmarkNum < 10) {
        cerr << fmt::format("lastCost.usefulLandmarkNum: {} too small!!!\n",
                            lastCost.usefulLandmarkNum);
        return false;
    }

    ResidualInfo firstCost = lastCost;
    chrono::steady_clock::time_point T1 = chrono::steady_clock::now();
    int continousNoImprovementNum = 0;
    for (int i = 0; i < maxIte_; ++i) {
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

        ConstructJ_H_b_g(i == 0);

        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();

        Eigen::VectorXd _lambda(H_.rows());
        _lambda.setConstant(lambda_);
        if (margKFstatus_) {
            if (i == 0) {
                cout << "Hp_[6x6]: " << setprecision(2)
                     << Hp_.diagonal().head(6).transpose() << endl;
                cout << fmt::format(
                    "H_ size:[{}x{}], Hp_ size: [{}x{}], g_ len: {}, g_p_ len: "
                    "{}\n",
                    H_.rows(), H_.cols(), Hp_.rows(), Hp_.cols(), g_.rows(),
                    g_p_.rows());
                cout << "Prior Message Added!!!" << endl;
            }

            H_ += Hp_;
            g_ += g_p_;
        } else {
            _lambda.head(6).setConstant(1e10);  // 首帧的约束足够大
            //_lambda.head(window_.size() * window_[0]->Tcw_.Size())
            //    .setConstant(DBL_MAX);  // 不优化位姿
            if (i == 0) {
                cout << "Fixed First Frame!!!" << endl;
            }
        }
        H_.diagonal() += _lambda;

        Eigen::VectorXd delta_x;
        if (!onlyPoseUpdate_) {
            chrono::steady_clock::time_point t3 = chrono::steady_clock::now();
            delta_x = SchurCompleteSolve(
                H_, g_, window_.size(), lastCost.usefulLandmarkNum,
                window_[0]->Twc_.Size(), optLandmark_[0]->Size());
            chrono::steady_clock::time_point t4 = chrono::steady_clock::now();
            if (i == 0) {
                cout << "SchurCompleteSolve spend: "
                     << chrono::duration<double>(t4 - t3).count() << " sec."
                     << endl;
            }
        } else {
            delta_x = H_.ldlt().solve(g_);
        }

        // 保留状态备份
        for (size_t i = 0; i < optLandmark_.size(); ++i) {
            if (!optLandmark_[i]->NoUsed() && optLandmark_[i]->initialized_) {
                optLandmark_[i]->CopyStatus();
            }
        }

        for (size_t i = 0; i < window_.size(); ++i) {
            window_[i]->CopyStatus();
        }

        // 状态更新
        UpdateStatusVariables(delta_x);
        // 判断当前更新是否有效，在使用新的pose计算cost时，可能会让一些点被设置为noUsed
        ResidualInfo newCost = CalculateResidualWindow();
        if (margKFstatus_) {
            // 需要考虑先验残差约束
            newCost.priorConstraintChi2 = CalculatePriorCost(delta_x);
        }
        newCost.cost += newCost.priorConstraintChi2;

        if (newCost.usefulLandmarkNum != lastCost.usefulLandmarkNum && 0) {
            // TODO：这里需要使用更新前的pose
            cout << "[WARNING]: "
                 << "recalculate last cost, last usefulNUm, new usefulNum: "
                 << lastCost.cost << ", " << lastCost.usefulLandmarkNum << ", "
                 << newCost.usefulLandmarkNum << endl;
            // TODO: 重新计算时，需要使用旧的poses，旧的landmarks位置，以及新的landmarks的noUsed标志，比较麻烦
            lastCost = CalculateResidualWindow(true);
            cout << "new usefulNum: " << lastCost.usefulLandmarkNum << endl;
        }

        if (config->iterateLogFreqLM > 0 && i % config->iterateLogFreqLM == 0) {
            cout << fmt::format(
                "Window BA iterate {} times, lastCost: {:.1f}, newCost: "
                "{:.1f}, useful lk num: {}, "
                "lambda: {}.\n",
                i, lastCost.cost, newCost.cost, newCost.usefulLandmarkNum,
                lambda_);

            cout << "ConstructJ_H_b_g spend: "
                 << chrono::duration<double>(t2 - t1).count() << " sec."
                 << endl;

            chrono::steady_clock::time_point t5 = chrono::steady_clock::now();
            cout << "LM one iteration spend: "
                 << chrono::duration<double>(t5 - t1).count() << " sec.\n"
                 << endl;
        }

        // 使用LM方法，考虑存在由于图像模糊投影不上的问题，因此newCost不能小于0
        bool accept = false;
        double costRelativeAbsDiff = 100;
        const double predictReduction =
            ComputePredictionReduction(delta_x, g_, H_);
        UpdateLMlambda(lastCost, newCost, predictReduction, accept,
                       continousNoImprovementNum, costRelativeAbsDiff);
        if (!accept) {
            for (size_t i = 0; i < optLandmark_.size(); ++i) {
                if (!optLandmark_[i]->NoUsed()) {
                    optLandmark_[i]->BackUpStatus();
                }
            }
            for (size_t i = 0; i < window_.size(); ++i) {
                window_[i]->BackUpStatus();
            }
        } else {
            lastCost = newCost;
            // 更新先验残差构成信息项
            if (margKFstatus_) {
                UpdatePriorDeltaX0(delta_x);
            }
        }

        if (LMstopJudge(continousNoImprovementNum, costRelativeAbsDiff,
                        delta_x)) {
            break;
        }
    }
    chrono::steady_clock::time_point T2 = chrono::steady_clock::now();
    const double spendTime = chrono::duration<double>(T2 - T1).count();
    cout << fmt::format(
        "First cost: {:.1f}, final cost: {:.1f}, first mean proj cost: {:.1f}, "
        "last mean proj cost: {:.1f}, priorConstraintChi2: {:.1f}, "
        "cost decrease ratio: {:.1f}%, usefulNum ratio: {:.1f}%, total "
        "optimize "
        "spend: {:.1f}s in window\n",
        firstCost.cost, lastCost.cost, firstCost.meanCost, lastCost.meanCost,
        lastCost.priorConstraintChi2,
        ((firstCost.cost - lastCost.cost) / firstCost.cost) * 100,
        double(lastCost.usefulLandmarkNum) / optLandmark_.size() * 100,
        spendTime);

    return lastCost.cost < firstCost.cost;
}

void Optimizer::PreSelectLandmarkForTracking(
    KeyFrame::OpticalFlowStruct& optFlw, std::vector<Landmark*>& lk1s,
    std::vector<Eigen::Vector2d>& obvs) {
    lk1s.reserve(optFlw.trackLandmark_.size());
    obvs.reserve(lk1s.size());
    constexpr int kDebugNum = 20000;

    auto SelectLandmark = [&lk1s, &obvs](const vector<Landmark*>& trackLandmark,
                                         const vector<cv::Point2f>& prevPts) {
        for (size_t i = 0; i < trackLandmark.size(); ++i) {
            // TODO: FEJ指的是关于逆深度的线性化点在首次计算出逆深度值时
            Landmark* lk = trackLandmark[i];
            if (lk->CanBeUseForOptimization()) {
                lk->ResetFEJ();
                lk1s.emplace_back(lk);
                const cv::Point2f& p = prevPts[i];
                obvs.emplace_back(p.x, p.y);
#if defined(WRITE_MATCH_PAIR_IMAGE)
                //DrawProjectCase(*lk, obvs.back(), optFlw.prevImg_, Twc2);
#endif
            }
            if (lk1s.size() > kDebugNum) {
                break;
            }
        }
    };

    SelectLandmark(optFlw.trackLandmark_, optFlw.prevPts_);
    SelectLandmark(optFlw.trackHistoryLandmark_, optFlw.prevHistoryPts_);

    cout << fmt::format(
        "curF pose opt preselect landmark num: {}, last kf track feature num: "
        "{}, "
        "history kf track feature num: {}\n",
        lk1s.size(), optFlw.trackLandmark_.size(),
        optFlw.trackHistoryLandmark_.size());
}

bool Optimizer::OptimizeCurFrame(KeyFrame::OpticalFlowStruct& optFlw,
                                 Pose& Twc2, const int curFid,
                                 int& totalPointNum, int& usefulPointNum) {

    // 构建优化问题所需观测
    vector<Landmark*> preLks;
    vector<Eigen::Vector2d> preObvs;
    PreSelectLandmarkForTracking(optFlw, preLks, preObvs);
    totalPointNum = preLks.size();

    if (preLks.empty()) {
        cout << "useful landmark num for opt is: " << preLks.size()
             << " Error! may be no initialized???\n";
        //exit(-1);
        return false;
    }

#if defined(WRITE_MATCH_PAIR_IMAGE)
    //WriteDebugTriangulateCase2Video(curFid);
#endif
    double initLambda = lambda_;
    lambda_ = initLambda;
    vector<Landmark*> stableLks;
    vector<Eigen::Vector2d> stableObvs;
    ResidualInfo lastCost =
        SetOptimizeLandmarkForTracking(preLks, preObvs, Twc2, optFlw.prevImg_,
                                       usefulPointNum, stableLks, stableObvs);

    if (stableLks.size() < 100) {
        cout << "use " << stableLks.size() << " landmarks to optimize!\n";
    }

    ResidualInfo firstCost = lastCost;
    if (firstCost.usefulLandmarkNum < 20) {
        cout << fmt::format("Error first useful constrint num: {}\n",
                            firstCost.usefulLandmarkNum);
        return false;
    }

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    int continousNoImprovementNum = 0;
    for (int i = 0; i < maxIte_; ++i) {
        CalculateHandGradiantCurFrame(stableLks, stableObvs, Twc2, H_, g_);
        if (lastCost.usefulLandmarkNum < 20) {
            cout << fmt::format("Error useful constrint num: {}\n",
                                lastCost.usefulLandmarkNum);
            return false;
        }
        Eigen::VectorXd _lambda(H_.rows());
        _lambda.setConstant(lambda_);
        if (H_.diagonal().head(6).isApproxToConstant(0)) {
            _lambda.head(6).setConstant(0);
        }

        // debug, 返回J, 判断H, g计算的正确性
        H_.diagonal() += _lambda;
        //cout << setprecision(5) << "H_:\n " << H_.diagonal().transpose()
        //     << endl;
        //cout << setprecision(5) << "g_:\n " << g_.transpose() << endl;
        Eigen::VectorXd delta_x = H_.ldlt().solve(g_);

        const Pose poseBackup = Twc2;

        // 当前帧pose状态更新
        const int startRow = 0;
        Twc2.Update(delta_x.middleRows(startRow, 3),
                    delta_x.middleRows(startRow + 3, 3));

        // 判断当前更新是否有效
        ResidualInfo newCost =
            CalculateResidualCurFrame(stableLks, stableObvs, Twc2);

        if (config->iterateLogFreqLM > 0 && i % config->iterateLogFreqLM == 0) {
            cout << setprecision(5) << "delta_pose: " << delta_x.transpose()
                 << "\n";
            cout << fmt::format(
                "CurF BA iterate {} times, lastCost: {:.1f}, newCost: {:.1f}, "
                "useful landmark: {}, "
                "lambda: {}.\n",
                i, lastCost.cost, newCost.cost, newCost.usefulLandmarkNum,
                lambda_);
        }

        bool accept = false;
        double costRelativeAbsDiff = 100;
        const double predictReduction =
            ComputePredictionReduction(delta_x, g_, H_);
        UpdateLMlambda(lastCost, newCost, predictReduction, accept,
                       continousNoImprovementNum, costRelativeAbsDiff);
        // LM 方法
        if (!accept) {
            Twc2 = poseBackup;
        } else {
            lastCost = newCost;
        }

        if (LMstopJudge(continousNoImprovementNum, costRelativeAbsDiff,
                        delta_x)) {
            break;
        }
    }
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    const double spendTime = chrono::duration<double>(t2 - t1).count();
    cout << fmt::format(
        "First cost: {:.1f}, final cost: {:.1f}, first mean proj cost: {:.1f}, "
        "last mean proj cost: {:.1f}, "
        "cost decrease ratio: {:.1f}%, usefulNum ratio: {:.1f}%, total "
        "optimize "
        "spend: {:.3f}s.\n",
        firstCost.cost, lastCost.cost, firstCost.meanCost, lastCost.meanCost,
        ((firstCost.cost - lastCost.cost) / firstCost.cost) * 100,
        double(lastCost.usefulLandmarkNum) / stableLks.size() * 100, spendTime);

    return lastCost.cost < firstCost.cost;
}

void Optimizer::AddOneKeyFeame(KeyFrame* kf) {
    // TODO: 上一帧需要添加对当前KF的观测，此外，
    // 如果上一帧的landmark继续被下一帧看到，应该也要实现持续跟踪的效果
    // 考虑是否需要转移landmark所有权？
    // 直接在新KF中提取关键点，并放入optFlw结构中，同时保留上一KF的跟踪结果仍进行跟踪
    KeyFrame* lastKf = window_.empty() ? nullptr : window_.back();
    const int totalTrackLandmarkNum = kf->InitializeLandmark(lastKf);
    cout << fmt::format("kf id: {}, optical flow total feature num: {}\n",
                        kf->id_, totalTrackLandmarkNum);
    int historyTriSucceedNum = 0;
    int prevTriSucceedNum = 0;

    KeyFrame::OpticalFlowStruct& optFlw = KeyFrame::optFlw;
    if (!window_.empty()) {
        unordered_map<KeyFrame*, Pose> kf2T12;
        for (size_t i = 0; i < optFlw.trackHistoryLandmark_.size(); ++i) {
            Landmark* lk = optFlw.trackHistoryLandmark_[i];
            // 仍被当前帧观测到，可以进行深度滤波更新，或者进行多视角优化
            const cv::Point2f& p = optFlw.prevHistoryPts_[i];  // curFrameObv
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
                    ++lk->failInitializeNum_;
                    continue;
                }
                lk->SetTriangulateResult(idepth1);
                ++historyTriSucceedNum;
#if defined(WRITE_MATCH_PAIR_IMAGE)
//DrawTriangulateCase(idepth1, *lk, curObv.cast<int>(),
//                    kf->debugGrayImg_, T12);
#endif
            }
        }

        for (size_t i = 0; i < optFlw.prevPts_.size(); ++i) {
            Landmark* lk = optFlw.trackLandmark_[i];
            const cv::Point2f& p = optFlw.prevPts_[i];  // curFrameObv
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

#if defined(WRITE_MATCH_PAIR_IMAGE)
                //DrawTriangulateCase(idepth1, *lk, curObv.cast<int>(),
                //                    kf->debugGrayImg_, T12);
#endif
                lk->SetTriangulateResult(idepth1);
                ++prevTriSucceedNum;
            }
        }

        int removeFeatNum = window_.back()->RemoveNoInitializeLongFeature();
        cout << fmt::format(
            "prevTriSucceedNum: {}, historyTriSucceedNum: {}, remove long time "
            "fail initialize feature num: {}\n",
            prevTriSucceedNum, historyTriSucceedNum, removeFeatNum);

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

        // 滑窗优化时，会将当前帧添加到滑窗中去
        if (SlidingWindowOptimize(kf)) {
            //exit(0);
        }
        //window_.push_back(kf);
    } else {
        // 创建的是首帧关键帧，是否需要赋值prevHistoryPts_？
        // 应该是不需要的
        window_.push_back(kf);
    }
    // TODO 1：进行滑窗BA
    // TODO 2：进行三角化工作

    cout << "add kf id: " << kf->id_ << "\n";
    cout << fmt::format(
        "Triangulate by KF_{} report: trackHistoryLandmark size: {}, "
        "historyTriSucceedNum: {}, "
        "prev trackLandmark size: {}, prevTriSucceedNum: {}\n",
        kf->id_, optFlw.trackHistoryLandmark_.size(), historyTriSucceedNum,
        optFlw.trackLandmark_.size(), prevTriSucceedNum);
}

void Optimizer::UpdateStatusVariables(const Eigen::VectorXd& deltaX,
                                      const size_t startPoseId) {
    int updateId = 0;
    for (size_t i = startPoseId; i < window_.size(); ++i) {
        const int startRow = (i - startPoseId) * window_[0]->Twc_.Size();
        window_[i]->Update(deltaX.middleRows(startRow, 3),
                           deltaX.middleRows(startRow + 3, 3));
    }
    if (!onlyPoseUpdate_) {
        updateId += (window_.size() - startPoseId) * window_[0]->Twc_.Size();
        for (size_t i = 0; i < optLandmark_.size(); ++i) {
            if (!optLandmark_[i]->NoUsed()) {
                optLandmark_[i]->Update(
                    deltaX.middleRows(updateId, optLandmark_[0]->Size())[0]);
                ++updateId;
            }
        }
    }
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

bool Optimizer::TransformLandmarkOwnerFromOldestKF(const int margKFid) {
    if (margKFid < 0) {
        return false;
    }

    // 在选择边缘化帧时已经把需要移除的帧移动到了开头
    KeyFrame* oldest = window_[0];
    cout << fmt::format("window size: {}, begin remove kf id: {}\n",
                        window_.size(), oldest->id_);

    KeyFrame* nextKF = window_[margKFid + 1];
    int transformLandmarkNum = 0;
    int newAddOptimizeLandmarkNum = 0;
    for (Landmark* lk : oldest->landmark_) {
        bool transformSucceed = false;
        for (auto kf2lk : lk->target_) {
            KeyFrame* kf = kf2lk.first;
            if (nextKF != kf) {
                continue;
            }

            // 把地图点的所有权转移到最新KF，其余的不要
            // 这里需要将lk从oldestKF中删除，并将其添加到下一个KF，且需要保持地址不变
            if (lk->TransformHost2OtherKF(nextKF)) {
                ++transformLandmarkNum;
                transformSucceed = true;
                // 由于先前没有添加待删除关键帧的地图点至优化变量，
                // 这里将被保留下来的地图点添加进来以进行滑窗BA
                // 后续会删除关于oldest KF的观测
                if (lk->initialized_ &&
                    lk->target_.size() > kMinUsefulObvNumWithHost) {
                    optLandmark_.emplace_back(lk);
                    ++newAddOptimizeLandmarkNum;
                }
            }
            break;
        }
        if (!transformSucceed) {
            lk->SetCanDelete();
        }
    }
    cout << fmt::format(
        "margKF transform landmark num: {}, newAddOptimizeLandmarkNum: {}, "
        "total landmark num: {}\n",
        transformLandmarkNum, newAddOptimizeLandmarkNum,
        oldest->landmark_.size());
    return true;
}

void Optimizer::RemoveOldestKeyFrame(const int margKFid) {
    if (margKFid < 0) {
        return;
    }

    KeyFrame* oldest = window_[0];
    window_.erase(window_.begin());

    // 注意：此时应该已经完成边缘化时的先验信息构建操作
    // 移除掉边缘化帧对地图点的观测
    for (const KeyFrame* kf : window_) {
        for (Landmark* lk : kf->landmark_) {
            if (lk->target_.count(oldest)) {
                lk->target_.erase(oldest);
            }
        }
    }
    // 移除光流跟踪中被标记为可以删除的Landmark
    auto RemoveDeleteLandmarkFromOpticalFlow =
        [](vector<Landmark*>& lks, vector<cv::Point2f>& obvs) -> void {
        vector<Landmark*>::iterator it1 = lks.begin();
        vector<cv::Point2f>::iterator it2 = obvs.begin();
        while (it1 != lks.end()) {
            if ((*it1)->CanBeDelete()) {
                it1 = lks.erase(it1);
                it2 = obvs.erase(it2);
                continue;
            }
            ++it1;
            ++it2;
        }
    };
    KeyFrame::OpticalFlowStruct& optFlw = KeyFrame::optFlw;
    RemoveDeleteLandmarkFromOpticalFlow(optFlw.trackLandmark_, optFlw.prevPts_);
    RemoveDeleteLandmarkFromOpticalFlow(optFlw.trackHistoryLandmark_,
                                        optFlw.prevHistoryPts_);

    // 删除老帧看看是否会有影响
    // delete oldest;
    delayEraseKeyframe_.push_back(oldest);
    if (delayEraseKeyframe_.size() > 1) {
        lock_guard<std::mutex> lock(KeyFrame::mutexForSyncView3Dstatus);
        if (!KeyFrame::kfOn3Dshow.count(delayEraseKeyframe_[0])) {
            delete delayEraseKeyframe_[0];
            delayEraseKeyframe_[0] = nullptr;
            delayEraseKeyframe_.erase(delayEraseKeyframe_.begin());
        }
    }
    return;
}

int Optimizer::SampleUsefulLandmark(const int margKFid) {
    // 此时还未添加最新关键帧，并且最老关键帧已经选出来，且移到第0位
    const int startKFid = margKFid < 0 ? 0 : 1;
    optLandmark_.clear();
    for (int i = startKFid; i < static_cast<int>(window_.size()); ++i) {
        for (Landmark* p : window_[i]->landmark_) {
            // 地图点有被其他关键帧看到
            if (!p->initialized_ || p->IsOutOfRange() || p->CanBeDelete() ||
                p->target_.size() < 3) {
                continue;
            }
            p->ResetFEJ();
            optLandmark_.emplace_back(p);
        }
    }

    return optLandmark_.size();
}

Optimizer::ResidualInfo Optimizer::CalculateResidualWindow(
    const bool useBackUpStatus) {
    ResidualInfo info;

    for (size_t i = 0; i < optLandmark_.size(); ++i) {
        Landmark*& p = optLandmark_[i];
        if (p->NoUsed()) {
            continue;
        } else {
            ++info.usefulLandmarkNum;
        }

        KeyFrame* host = p->host_;
        const Eigen::Vector3d pc1 = p->GetPc(useBackUpStatus);
        const Eigen::Vector3d pw =
            useBackUpStatus ? host->TwcBack_ * pc1 : host->Twc_ * pc1;

        for (const auto& kf2obv : p->target_) {
            KeyFrame* target = kf2obv.first;
            // 这里如果还未被标记为noUsed_，那么会多计算几个残差，
            // 但是影响不大，可能会出现
            if (target == host) {
                continue;
            }

            const Eigen::Vector3d pc2 =
                useBackUpStatus ? target->TcwBack_ * pw : target->Tcw_ * pw;
            const Eigen::Vector2d px2 = cam_->Project2PixelPlane(pc2);

            Eigen::Vector2d r = px2 - kf2obv.second;
            double chi2 = r.squaredNorm();
            Eigen::Vector2d rho;
            HuberLoss(chi2, rho);
            info.cost += rho[0];
            ++info.totalConstraintNum;
        }
    }

    info.meanCost = info.cost / info.totalConstraintNum;
    // cout << fmt::format(
    //     "window BA: residual info.all.cost: {:.1f}, useful landmark num: {}, "
    //     "total constraint num: {}, mean cost: {:.1f}\n",
    //     info.cost, info.usefulLandmarkNum, info.totalConstraintNum,
    //     info.meanCost);

    if (isnan(info.cost) || isinf(info.cost)) {
        cout << fmt::format("Error window cost value: {}, landmark num: {}\n",
                            info.cost, info.usefulLandmarkNum);
        info.cost = DBL_MAX;
    }
    return info;
}

void Optimizer::DebugOptlandmarkStatus(const size_t num,
                                       const std::string& name) {
    string debugUsefulLkIndex(name + " useful lk status:[ ");
    for (size_t i = 0; i < min(optLandmark_.size(), num); ++i) {
        Landmark* p = optLandmark_[i];
        debugUsefulLkIndex.append(fmt::format(
            " {}-{}-{}", i, reinterpret_cast<size_t>(p), p->NoUsed()));
    }
    debugUsefulLkIndex.append(" ]");
    cout << debugUsefulLkIndex << endl;
}

Optimizer::ResidualInfo Optimizer::SetOptimizeStatusVariableForWindowBA(
    KeyFrame* const margKF) {
    ResidualInfo info;

    const double maxChi2 = config->maxProjectError * config->maxProjectError;
    for (size_t i = 0; i < optLandmark_.size(); ++i) {
        Landmark*& p = optLandmark_[i];
        if (p->NoUsed()) {
            continue;
        }

        if (!p->CanBeUseForOptimization()) {
            p->SetNoUsed();
            continue;
        }

        KeyFrame* host = p->host_;
        if (host->id_ == window_.back()->id_) {
            p->SetNoUsed();
            continue;
        }

        ResidualInfo tempInfo;
        const Eigen::Vector3d pc1 = p->GetPc();
        const Eigen::Vector3d pw = host->Twc_ * pc1;

        for (const auto& kf2obv : p->target_) {
            KeyFrame* target = kf2obv.first;
            // 这里如果还未被标记为noUsed_，那么会多计算几个残差，
            // 但是影响不大，可能会出现
            if (target == host) {
                continue;
            }

            const Eigen::Vector3d pc2 = target->Tcw_ * pw;
            const Eigen::Vector2d px2 = cam_->Project2PixelPlane(pc2);

            const bool inRange = InRange(target->grayImg_, px2.cast<int>());
            if (!inRange || pc2.z() < kMinSceneDepthInCamera) {
                // 需要认识到的是，未更新前的{poses, landmarks}一定是可以投影成功的！！！
                // 所以理论上，这一步只会被更新后的{poses, landmarks}执行
                p->SetNoUsed();
                break;
            } else {
                Eigen::Vector2d r = px2 - kf2obv.second;
                double chi2 = r.squaredNorm();
                if (chi2 > maxChi2) {
                    // 必须移除残差异常的项
                    p->SetNoUsed();
                    break;
                }

                Eigen::Vector2d rho;
                HuberLoss(chi2, rho);
                tempInfo.cost += rho[0];
                ++tempInfo.totalConstraintNum;
            }
        }

        // 只有未被其余观测标记异常的地图点才可参与残差构建
        // 边缘化时，由于会移除边缘化帧，因此其观测不可计入
        const int excludeNum =
            (margKF != nullptr && p->target_.count(margKF)) ? 1 : 0;
        if (!p->NoUsed() &&
            tempInfo.totalConstraintNum >= (kMinUsefulObvNum + excludeNum)) {
            info.cost += tempInfo.cost;
            info.totalConstraintNum += tempInfo.totalConstraintNum;
            ++info.usefulLandmarkNum;
        } else {
            p->SetNoUsed();
        }
    }

    info.meanCost = info.cost / info.totalConstraintNum;
    cout << fmt::format(
        "window BA: set opt variable residual info.all.cost: {:.1f}, useful "
        "landmark num: {}, "
        "total constraint num: {}, mean cost: {:.1f}\n",
        info.cost, info.usefulLandmarkNum, info.totalConstraintNum,
        info.meanCost);

    if (isnan(info.cost) || isinf(info.cost)) {
        cout << fmt::format("Error window cost value: {}, landmark num: {}\n",
                            info.cost, info.usefulLandmarkNum);
        info.cost = DBL_MAX;
    }
    return info;
}

// TODO： 先不考虑边缘化，而是直接丢弃首帧
bool Optimizer::MarginalizeOldestKeyFrame() {
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
    //set<Landmark*> margLandmark;
    // TODO: 选择另一种策略移除一帧，类似Landmark的处理方式，将其移到window_[0]再构建信息矩阵H即可
    //KeyFrame* margKF = window_[0];
    // 边缘化地图点会破坏H_矩阵的稀疏性，通过实现地图点控制权转移来避免删除点
    //for (size_t i = 0; i < optLandmark_.size(); ++i) {
    //    Landmark* p = optLandmark_[i];

    //    if (p->host_->id_ == margKF->id_) {
    //        margLandmark.insert(p);
    //    }
    //}

    //vector<Landmark*> sortMargOptLandmark;
    //for (Landmark* p : margLandmark) {
    //    // OK, 这样就实现了将边缘化地图点移到左上角的目的啦！！！
    //    sortMargOptLandmark.push_back(p);
    //}
    //for (Landmark* p : optLandmark_) {
    //    if (!margLandmark.count(p)) {
    //        sortMargOptLandmark.push_back(p);
    //    }
    //}
    // 根据新的Landmark顺序，构建信息矩阵H_，并保存FEJ
    //optLandmark_ = sortMargOptLandmark;
    //cout << "total, marg, left landmars: " << optLandmark_.size() << " "
    //     << margLandmark.size() << " "
    //     << (optLandmark_.size() - margLandmark.size()) << endl;

    SetOptimizeStatusVariableForWindowBA(window_[0]);
    ConstructJ_H_b_g(true);

    // Step:接下来计算相关先验Hp, g_p
    const int poseDim = window_[0]->Twc_.Size();
    constexpr int depthDim = 1;
    //const int margDim = poseDim + margLandmark.size() * depthDim;
    // 这里我们直接将最老帧的landmark转移或丢弃不用，只保留边缘化最老帧的信息
    const int margDim = poseDim;
    const int leftDim = H_.cols() - margDim;
    // 使用舒尔补进行边缘化H矩阵，并形成上三角矩阵
    // | I          0 |   | A  B |   | A  B |
    // | -C*A.inv   I | * | C  D | = | 0  ΔA| ==> ΔA = -C*A.inv*B + D
    Eigen::MatrixXd A = H_.block(0, 0, margDim, margDim);
    const double kMinAmatrixDet = 1e-16;
    if (abs(A.determinant()) < kMinAmatrixDet) {
        // 边缘化信息过小，无效
        return false;
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
    // | A  B  |   |x1|   | I          0 |   |g1|
    // | 0  ΔA | * |x2| = | -C*A.inv   I | * |g2| ==>
    // TODO: 留下来的状态量X2如果更新，右边的先验残差怎么变呢?
    g_p_ = temp * g_.head(margDim) + g_.tail(leftDim);
    cout << "Marginalization info:\nA: " << setprecision(2)
         << A.diagonal().transpose() << "\n"
         << "invA: " << invA.diagonal().transpose() << "\n"
         << "temp: " << temp.diagonal().transpose() << "\n"
         << "B: " << B.diagonal().transpose() << "\n"
         << "D: " << D.diagonal().head(12).transpose() << "\n"
         << "Hp_[6x6]:\n"
         << Hp_.block(0, 0, 6, 6) << "\n"
         << "gp_: " << g_p_.transpose() << "\n";
    // 构建先验增量，边缘化帧改变了原有的概率分布，需要将该增量用于更新状态量
    const Eigen::VectorXd deltaX = SchurCompleteSolve(
        Hp_, g_p_, config->maxKFnumInWindow,
        Hp_.cols() - config->maxKFnumInWindow * config->maxKFnumInWindow,
        window_[0]->Tcw_.Size(), optLandmark_[0]->Size(), false);
    cout << "marg KF delta x: " << deltaX.transpose() << endl;
    //if (!CalculatePriorCostChi2(deltaX)) {
    //    cout << fmt::format(
    //        "calculate prior chi2 failed, marg kf id: {}, new kf id: {}, will "
    //        "keep first kf in window fixed!\n",
    //        window_[0]->id_, window_.back()->id_);
    //    return false;
    //}
    UpdateStatusVariables(deltaX, 1);

    // 重置记录增量的变量
    margDeltaX_.resize(deltaX.size());
    margDeltaX_.setZero();
    return true;

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

    // 暂时不边缘化点，而是转移到邻近帧或者直接丢弃
    //if(!margLandmark.empty()) {
    //    // 构建完H矩阵后，可以从优化地图点中移除marg landmark
    //    optLandmark_.erase(optLandmark_.begin(), optLandmark_.begin() + margLandmark.size());
    //}
}

bool Optimizer::CalculatePriorCostChi2(const Eigen::VectorXd& deltaX) {
    // 1. 特征值分解 H_p
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(Hp_);
    if (eigensolver.info() != Eigen::Success) {
        cerr << "Marginalize Eigen decomposition Hp_ failed!. Fixed first "
                "frame instead!\n";
        return false;
    }

    Eigen::VectorXd eigenvalues = eigensolver.eigenvalues();
    Eigen::MatrixXd eigenvectors = eigensolver.eigenvectors();
    constexpr double kMinEigenValue = 1e-20;
    for (int i = 0; i < eigenvalues.rows(); ++i) {
        // 一般只发生在边缘化上一个最新帧的情况
        if (eigenvalues[i] < kMinEigenValue) {
            cout << fmt::format(
                "Hp_ eigen value {} is: {}, total eigen value num: {}\n", i,
                eigenvalues[i], eigenvalues.rows());
            return false;
        }
    }

    // 2. 构造 J_p = sqrt(Λ) * V^T
    Eigen::VectorXd sqrt_eigenvalues = eigenvalues.cwiseSqrt();
    Eigen::MatrixXd J_p =
        sqrt_eigenvalues.asDiagonal() * eigenvectors.transpose();

    // 根据先验方程等式，有：
    const Eigen::VectorXd priorResidual = -J_p * deltaX;
    // 该项是常数项，不需要考虑
    rpChi2_ = priorResidual.squaredNorm();
    cout << fmt::format("prior residual chi2: {}!\n", rpChi2_);

    return true;
}

double Optimizer::ComputePredictionReduction(const Eigen::VectorXd& deltaX,
                                             const Eigen::VectorXd& g,
                                             const Eigen::MatrixXd& H) {
    // 实际下降值为： lastCost - newCost
    // g = -J.T * r
    return -0.5 * deltaX.dot(H * deltaX) + deltaX.dot(g) -
           0.5 * lambda_ * deltaX.squaredNorm();
}

void Optimizer::UpdateLMlambda(const Optimizer::ResidualInfo& lastCost,
                               const Optimizer::ResidualInfo& newCost,
                               const double predictReduction, bool& accept,
                               int& continousNoImprovementNum,
                               double& costRelativeAbsDiff) {
    // TODO: 使用更好的LM更新方式
    costRelativeAbsDiff = lastCost.cost - newCost.cost;
    const double rho = costRelativeAbsDiff / (predictReduction + 1e-12);
    if (rho > 0) {
        if (rho > 0.75) {
            lambda_ *= 0.3;
        }
        accept = true;
        continousNoImprovementNum = 0;
    } else {
        lambda_ *= 1.8;
        accept = false;
        ++continousNoImprovementNum;
    }
    costRelativeAbsDiff = abs(costRelativeAbsDiff);
}

bool Optimizer::LMstopJudge(const int& continousNoImprovementNum,
                            const double& costRelativeAbsDiff,
                            const Eigen::VectorXd& delta) {
    const bool lambdaTestEnough = lambda_ > 1e9 || lambda_ < 1e-9;
    if (costRelativeAbsDiff < config->convergeCostDiffLM && lambdaTestEnough) {
        cout << fmt::format("LM cost diff: {} converge!\n",
                            costRelativeAbsDiff);
        return true;
    }
    if (lambda_ > config->maxLambdaValueLM) {
        cout << fmt::format("lambad too large: {}\n", lambda_);
        return true;
    }
    if (continousNoImprovementNum > config->maxNoImprovementCountLM &&
        lambdaTestEnough) {
        cout << fmt::format("continousNoImprovementNum: {}, lambda_: {}\n",
                            continousNoImprovementNum, lambda_);
        return true;
    }
    if (delta.norm() < 1e-5 && lambdaTestEnough) {
        return true;
    }

    return false;
}

void Optimizer::ConstructJ_H_b_g(const bool logOut) {
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
    //const int variableDim =
    //    window_.size() * poseDim + optLandmark_.size() * depthDim;
    int variableDim = window_.size() * poseDim;
    for (const Landmark* lk : optLandmark_) {
        variableDim += lk->NoUsed() ? 0 : lk->Size();
    }
    if (logOut) {
        cout << "opt variable dim: " << variableDim << endl;
    }

    H_.resize(variableDim, variableDim);
    H_.setZero();
    g_.resize(variableDim);
    g_.setZero();

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
    int usefulLandmarkNum = 0;
    for (size_t i = 0; i < optLandmark_.size(); ++i) {
        Landmark* p = optLandmark_[i];
        if (p->NoUsed()) {
            continue;
        }

        // 注意，必须确保每个landmark都能构建残差以成为状态变量，否则优化变量位置会有问题，导致最终更新出错
        KeyFrame* host = p->host_;
        const Eigen::Vector3d pc1 = p->GetPc();
        const Eigen::Vector3d pw = host->Twc_ * pc1;

        // res w.r.t (u2, v2) [2x2]的单位矩阵
        // px2 w.r.t Pc2 [2x3]
        Eigen::Matrix<double, 2, 3> J_px2_Pc2Norm =
            p->cam_->K_[0].block(0, 0, 2, 3);
        J_px2_Pc2Norm(0, 2) = 0.;
        J_px2_Pc2Norm(1, 2) = 0.;

        // 最新关键帧没有反向追踪能力
        bool addConstraint = false;

        for (const auto& kf2obv : p->target_) {
            // 需要注意每个关键帧、每个landmark在H矩阵中的位置
            KeyFrame* target = kf2obv.first;
            if (target == host) {
                continue;
            }

            const Eigen::Vector3d pc2 = target->Tcw_ * pw;
            const Eigen::Vector2d px2 = cam_->Project2PixelPlane(pc2);

            // 经过校验，可以构建residual和jacobian
            const Eigen::Vector2d r = px2 - kf2obv.second;

            // 使用胡伯核函数剔除异常残差值
            double chi2 = r.squaredNorm();
            Eigen::Vector2d rho;
            HuberLoss(chi2, rho);

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
            J_Pc2Norm_Pc2 << d, 0, -pc2.x() * d2, 0, d, -pc2.y() * d2, 0, 0, 0;
            const Eigen::Matrix<double, 2, 3> J_px2_Pc2 =
                J_px2_Pc2Norm * J_Pc2Norm_Pc2;
            const Eigen::Matrix<double, 2, 3>& J_res_Pc2 = J_px2_Pc2;

            // Pc2 w.r.t Twc2 : Pc2 = Twc2.inv * Pw
            Eigen::Matrix<double, 3, 6>
                J_Pc2_Twc2;  // ------------------------> optimization variable
            const Eigen::Vector3d dt = pw - target->Twc_.t_wb_;
            // Pc2 w.r.t Rwc2
            J_Pc2_Twc2.block(0, 0, 3, 3) =
                SkewSymmetric(target->Tcw_.q_wb_ * dt);  // Twc_.q_wb_.inverse()
            // Pc2 w.r.t Pwc2
            J_Pc2_Twc2.block(0, 3, 3, 3) =
                -target->Tcw_.q_wb_
                     .toRotationMatrix();  // Twc.q_wb.R.transpose()

            // Pc2 w.r.t Pw
            // TODO: 这里也应该要使用首次的Tcw值吧！！！由于target有多帧，所以要保留多个
            const Eigen::Matrix3d J_Pc2_Pw =
                target->Tcw_.q_wb_.toRotationMatrix();

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

            // Residual w.r.t optimization variables Jacobian
            Eigen::Matrix<double, 2, 6> A1 =
                J_res_Pc2 * J_Pc2_Pw * J_Pw_Twc1;  // J_res_Pw * J_Pw_Twc1;
            if (host == window_[0]) {
                // fixed滑动窗口第一帧，不在这里执行，而是添加大的lambda或使用先验约束其变化量
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
                      bj = depthStartCol + usefulLandmarkNum;
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

            addConstraint = true;
        }

        if (addConstraint) {
            ++usefulLandmarkNum;
        }
    }
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

bool Optimizer::SlidingWindowOptimize(KeyFrame* curKF) {
    const int margKFid = SelectOneKF2Marginalization(*curKF);
    const int sampleNum = SampleUsefulLandmark(margKFid);
    // 在滑窗优化前就把最新KF添加到滑窗之中
    window_.emplace_back(curKF);
    if (window_.size() < 3) {
        return false;
    }
    cout << fmt::format(
        "Begin SlidingWindowOptimize! margKFid: {}, Sample landmark num for "
        "window BA: {}.\n",
        margKFid, sampleNum);

    margKFstatus_ = false;
    if (TransformLandmarkOwnerFromOldestKF(margKFid)) {
        // 只需要保留最老帧的信息即可，或者只固定首帧的pose进行优化在debug阶段也是可取的
        // 其信息已经通过深度点的传播转移到后面的KF中
        if (margKFid < 2 && config->useMarginalization) {
            margKFstatus_ = MarginalizeOldestKeyFrame();
            cout << fmt::format("marg kf succeed: {}\n", margKFstatus_);
        }

        // 如果是使用点-点匹配逻辑的话，那么应该先进行边缘化再转移点的控制权
        // 产生的问题是：那些没有host被边缘化，但是没有target的点不造成影响
        // 那些host被边缘化，但是仍有target的点，可能只剩一个target本身的观测
        RemoveOldestKeyFrame(margKFid);
    }

    SetInitLambda(1.0);
    chrono::steady_clock::time_point t0 = chrono::steady_clock::now();
    const bool winOptSuccess = ExecuteWindowOptimize();
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    const double spendTime = chrono::duration<double>(t1 - t0).count();
    cout << fmt::format("win size: {}, win BA spend {} sec!\n", window_.size(),
                        spendTime);

    const int markDeleteNum = MarkBigResidualLandmarkDelete();
    cout << fmt::format("markDeleteNum: {}, winOptSuccess: {}\n", markDeleteNum, winOptSuccess);

    return winOptSuccess;
}

void Optimizer::HuberLoss(const double chi2, Eigen::Vector2d& rho) {

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
        rho[0] = 0.5 * chi2;
        rho[1] = 1.0;
    }
}

int Optimizer::SelectOneKF2Marginalization(const KeyFrame& curKF) {
    // 考虑到还要把curKF加进来，因此这里不取等号
    if (static_cast<int>(window_.size()) < config->maxKFnumInWindow) {
        return -1;
    }

    // 说明预设关键帧数量较少，直接移除最老帧
    int smallId = 0;
    if (window_.size() > 3) {
        const Eigen::Vector3d posDiff =
            curKF.Twc_.t_wb_ - window_[window_.size() - 2]->Twc_.t_wb_;
        const double horDist = posDiff.head(2).norm();
        cout << fmt::format("curKF hor dist to the last 2 frame: {:.2f}\n",
                            horDist);
        if (horDist < config->needNewKFtrans * 1.5) {
            smallId = window_.size() - 1;  // 移除掉最新的，因为重复观测可能大
            cout << "Will remove the last keyframe from window!\n";
        }
    }

    // 将待删除的最老帧移到滑窗开头
    KeyFrame* oldest = window_[smallId];
    window_.erase(window_.begin() + smallId);
    window_.insert(window_.begin(), oldest);
    return smallId;
}

bool Optimizer::TrackLocalMap(KeyFrame* kf2, bool& needNewKFbySight) {

    Pose Twc2 = kf2->Twc_;
    // 仅优化当前帧pose，避免由于其运动模糊影响landmark估计值导致系统崩溃
    // 同时加快计算速度
    int totalPointNum = 0;
    int usefulPointNum = 0;
    onlyPoseUpdate_ = true;
    const bool optSuccess = OptimizeCurFrame(KeyFrame::optFlw, Twc2, kf2->id_,
                                             totalPointNum, usefulPointNum);
    onlyPoseUpdate_ = false;
    const double usefulRatio = double(usefulPointNum) / totalPointNum;
    cout << fmt::format(
        "track totalPointNum: {}, usefulPointNum: {}, usefulRatio: {:.1f}\n",
        totalPointNum, usefulPointNum, usefulRatio);
    needNewKFbySight = usefulRatio < 0.3 || usefulPointNum < 30;

    Pose beforeTwc2 = kf2->Twc_;
    if (optSuccess) {
        kf2->SetTwc(Twc2);  // 关闭这个出现错乱，证明优化有效
    }
    cout << "cur frame pose diff: " << beforeTwc2.Inverse() * kf2->Twc_ << endl;
    return optSuccess;
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

int Optimizer::MarkBigResidualLandmarkDelete() {
    const double maxMeanChi2 = 6 * 6;
    int markCount = 0;
    for (size_t i = 0; i < optLandmark_.size(); ++i) {
        Landmark* lk = optLandmark_[i];
        if (lk->NoUsed()) {
            lk->SetCanDelete();
            continue;
        }
        KeyFrame* host = lk->host_;
        const Eigen::Vector3d pc1 = lk->GetPc();
        const Eigen::Vector3d pw = host->Twc_ * pc1;
        double sumChi2 = 0.;
        for (const auto& kf2obv : lk->target_) {
            KeyFrame* tar = kf2obv.first;
            if (tar == host) {
                continue;
            }

            const Eigen::Vector3d pc2 = tar->Tcw_ * pw;
            const Eigen::Vector2d px2 = cam_->Project2PixelPlane(pc2);

            Eigen::Vector2d r = px2 - kf2obv.second;
            double chi2 = r.squaredNorm();
            sumChi2 += chi2;
        }

        const double meanChi2 = sumChi2 / (lk->target_.size() - 1);
        if (meanChi2 > maxMeanChi2) {
            lk->SetCanDelete();
            ++markCount;
        }
    }
    return markCount;
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
    {
        lock_guard<std::mutex> lock(KeyFrame::mutexForSyncView3Dstatus);
        KeyFrame::kfOn3Dshow.clear();
        for (Landmark* p : optLandmark_) {
            if (p != nullptr && !aPoints.count(p) && !p->IsOutOfRange() &&
                p->ManySupport() && p->Converge()) {
                aPoints.insert(p);
            }
        }

        for (int i = 0; i < static_cast<int>(window_.size()); ++i) {
            // for(int i = window_.size()-1; i < window_.size(); ++i) {
            KeyFrame* kf = window_[i];
            for (Landmark* p : kf->landmark_) {
                if (p != nullptr && !aPoints.count(p) && !lPoints.count(p) &&
                    !p->CanBeDelete() && p->initialized_) {
                    lPoints.insert(p);
                }
            }

            KeyFrame::kfOn3Dshow.insert(kf);
        }
    }

    if (!aPoints.empty() || !lPoints.empty()) {
        ::ShowLocalMap(vTwc);
        // cout << "show " << vTwc.size() << " KFs " << (aPoints.size()+lPoints.size()) << " map points" << endl;
    } else {
        //cerr << "wait for local map..." << endl;
    }
}
