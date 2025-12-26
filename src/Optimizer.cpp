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
#include "ceres_problem.h"

using namespace std;
using namespace cv;

constexpr int kMinUsefulObvNum = 2;  // 扣除host的观测
constexpr int kMinUsefulObvNumWithHost = kMinUsefulObvNum + 1;
constexpr double kMaxSetError = 1e12;
constexpr double kMaxErrorRatio = 0.5;

const char* const kBeforeLoopClosurePoseFilePath =
    "./opt_before_loop_closure_pose.txt";
const char* const kClosurePoseFilePath = "./opt_loop_closure_pose.txt";

Optimizer::Optimizer(shared_ptr<Camera> cam, const double lambda,
                     const int maxIte, const bool onlyPoseUpdate)
    : onlyPoseUpdate_(onlyPoseUpdate), cam_(cam) {
    vecMargKf_.reserve(1e4);
}

Optimizer::~Optimizer() {
    if (debugTrackLostStatusVideoWriter_.isOpened()) {
        debugTrackLostStatusVideoWriter_.release();
    }

    // 排查内存泄漏
    for (KeyFrame* kf : window_) {
        if (kf != nullptr) {
            delete kf;
            kf = nullptr;
        }
    }
}

Optimizer::ResidualInfo Optimizer::CalculateResidualCurFrame(
    const vector<Eigen::Vector3d>& lk1s, const vector<Eigen::Vector2d>& obvs,
    const Pose& Twc2) {

    ResidualInfo info;
    string debugInfo("chi2 residuals: ");
    constexpr int kStepInfoOut = 300000;

    const Camera& cam = *cam_;
    const Pose Tc2w = Twc2.Inverse();
    for (size_t j = 0; j < lk1s.size(); ++j) {
        const Eigen::Vector3d pc = Tc2w * lk1s[j];
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
            debugInfo.append(
                fmt::format("chi2: {:.1f}-rho[0]: {:.1f}; ", chi2, rho[0]));
        }
    }

    //cout << fmt::format(
    //    "curF BA: residual info.all.cost: {:.1f}, useful landmark num: {}, "
    //    "total constraint num: {}, mean cost: {:.1f}\ndebug "
    //    "residual info: {}\n",
    //    info.cost, info.usefulLandmarkNum, info.totalConstraintNum, meanCost,
    //    debugInfo);
    if (isnan(info.cost) || isinf(info.cost) || info.totalConstraintNum == 0) {
        info.cost = DBL_MAX;
        info.meanCost = DBL_MAX;
    } else {
        info.meanCost = info.cost / info.totalConstraintNum;
    }
    return info;
}

Optimizer::ResidualInfo Optimizer::SetOptimizeLandmarkForTracking(
    const vector<shared_ptr<Landmark>>& lk1s,
    const vector<Eigen::Vector2d>& obvs, const Pose& Twc2, const cv::Mat& img,
    int& canUseNum, vector<Eigen::Vector3d>& stablePws,
    vector<Eigen::Vector2d>& stableObvs) {
    stablePws.clear();
    stableObvs.clear();
    stablePws.reserve(lk1s.size());
    stableObvs.reserve(lk1s.size());

    ResidualInfo info;
    string debugInfo("chi2 residuals: ");
    const int kStepInfoOut = lk1s.size() / 10;

    // 计算最大相对残差
    const Camera& cam = *cam_;
    const Pose Tc2w = Twc2.Inverse();
    vector<double> errorVec(lk1s.size(), 0);
    vector<double> usefulTrackDepth(lk1s.size(), 0.);
    lastKFmeanDepth_ = 0.;
    for (size_t i = 0; i < lk1s.size(); ++i) {
        const Eigen::Vector3d pc = Tc2w * lk1s[i]->GetPw(true);
        const Eigen::Vector2d px = cam.Project2PixelPlane(pc);
        const bool inRange = InRange(img, px.cast<int>());
        if (inRange && pc.z() > kMinSceneDepthInCamera) {
            errorVec[i] = (px - obvs[i]).squaredNorm();
            usefulTrackDepth[i] = pc.z();
        } else {
            errorVec[i] = kMaxSetError;
        }
    }
    vector<double> errorCopy = errorVec;
    sort(errorVec.begin(), errorVec.end());
    int usefulIdx = errorVec.size() - 1;
    while (errorVec[usefulIdx] == kMaxSetError) {
        --usefulIdx;
    }
    const double maxChi2 =
        max(config->maxProjectError * config->maxProjectError,
            errorVec[static_cast<int>(kMaxErrorRatio * usefulIdx)]);
    cout << fmt::format(
        "frame BA error range: [{:.1f}, {:.1f}], usefulIdx: {}, maxChi2: "
        "{:.1f}, "
        "errorVec.size: {}\n",
        errorVec.front(), errorVec.back(), usefulIdx, maxChi2, errorVec.size());
    // 根据观测数分，>1, >2, >3
    vector<vector<pair<int, double>>> resampleStableLkIndex2Chi2(3);
    chrono::steady_clock::time_point t0 = chrono::steady_clock::now();
    for (auto& vec : resampleStableLkIndex2Chi2) {
        vec.reserve(lk1s.size());
    }

    for (size_t j = 0; j < lk1s.size(); ++j) {
        shared_ptr<Landmark> lk1 = lk1s[j];
        const double& chi2 = errorCopy[j];
        if (chi2 > maxChi2) {
            continue;
        }

        Eigen::Vector2d rho;  // 残差值和核函数关于残差的导数
        HuberLoss(chi2, rho);
        if (j % kStepInfoOut == 0) {
            debugInfo.append(fmt::format(
                "curF set opt landmark chi2: {:.1f}-rho[0]: {:.1f}-kf id: "
                "{}-kp1:({:.0f}, "
                "{:.0f}); ",
                chi2, rho[0], lk1->host_->id_, lk1->GetHostFrameObv().x(),
                lk1->GetHostFrameObv().y()));
        }

        const int id = lk1->target_.size();
        switch (id) {
            case 2:
                // 观测数最少
                resampleStableLkIndex2Chi2[0].emplace_back(j, rho[0]);
                break;
            case 3:
                resampleStableLkIndex2Chi2[1].emplace_back(j, rho[0]);
                break;
            default:
                resampleStableLkIndex2Chi2[2].emplace_back(j, rho[0]);
                break;
        }
    }

    // 由观测数量由高到低进行采样
    const int stableLandmarkNum = resampleStableLkIndex2Chi2[1].size() +
                                  resampleStableLkIndex2Chi2[2].size();
    constexpr int kMaxSelectLandmarkNum = 200;  // 排查选点集中问题
    const int maxSampleLandmarkNum = stableLandmarkNum > kMaxSelectLandmarkNum
                                         ? int(stableLandmarkNum * 0.8)
                                         : kMaxSelectLandmarkNum;

    canUseNum = 0;  // 理论上有效的地图点数量，但不需要全部使用
    for (const auto& vec : resampleStableLkIndex2Chi2) {
        canUseNum += vec.size();
    }
    for (int i = 2; i >= 0; --i) {
        for (const pair<int, double>& idx2Chi2 :
             resampleStableLkIndex2Chi2[i]) {

            stablePws.emplace_back(lk1s[idx2Chi2.first]->GetPw(true));
            stableObvs.emplace_back(obvs[idx2Chi2.first]);
            info.cost += idx2Chi2.second;
            ++info.totalConstraintNum;
            ++info.usefulLandmarkNum;  // 这里每个地图点只会投影一次到当前帧
            lastKFmeanDepth_ += usefulTrackDepth[idx2Chi2.first];
            if (info.totalConstraintNum > maxSampleLandmarkNum) {
                break;
            }
        }

        if (info.totalConstraintNum > maxSampleLandmarkNum) {
            break;
        }
    }

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    const double spendTime = ChronoMillisecTimeDuration(t0, t1);

    if (isnan(info.cost) || isinf(info.cost) || info.totalConstraintNum == 0) {
        info.cost = DBL_MAX;
        lastKFmeanDepth_ = 0.;
    } else {
        info.meanCost = info.cost / info.totalConstraintNum;
        lastKFmeanDepth_ /= canUseNum;
    }
    cout << fmt::format(
        "curF BA: maxChi2: {:.1f}, canUse2TrackNum: {}, set opt tracking "
        "variable residual "
        "info.all.cost: {:.1f}, "
        "useful landmark num: {}, "
        "total constraint num: {}, mean cost: {:.1f}, spend time: "
        "{:.3f}ms\nlastKFmeanDepth_: {:.3f}, debug "
        "residual info: {}\n",
        maxChi2, canUseNum, info.cost, info.usefulLandmarkNum,
        info.totalConstraintNum, info.meanCost, spendTime, lastKFmeanDepth_,
        debugInfo);
    return info;
}

void Optimizer::CalculateHandGradiantCurFrame(
    const vector<Eigen::Vector3d>& lk1s, const vector<Eigen::Vector2d>& obvs,
    const Pose& Twc2, Eigen::Matrix<double, 6, 6>& H,
    Eigen::Matrix<double, 6, 1>& g) {
    constexpr int resDim = 2;
    H.setZero();
    g.setZero();

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
    Eigen::Matrix<double, 2, 3> J_px2_Pc2Norm = cam_->K_[0].block(0, 0, 2, 3);
    J_px2_Pc2Norm(0, 2) = 0.;
    J_px2_Pc2Norm(1, 2) = 0.;

    // Pc2 w.r.t T12 [3x6]
    Eigen::Matrix<double, 3, 6> J_Pc2_Twc2 =
        Eigen::Matrix<double, 3, 6>::Zero();
    // * Pc2 w.r.t t12
    J_Pc2_Twc2.block<3, 3>(0, 3) = -Tc2w.q_wb_.toRotationMatrix();

    Eigen::Matrix<double, 3, 3> J_Pc2Norm_Pc2;

    for (size_t j = 0; j < lk1s.size(); ++j) {
        const Eigen::Vector3d& Pw1 = lk1s[j];
        const Eigen::Vector3d Pc2 = Tc2w * Pw1;
        const Eigen::Vector2d px2 = cam_->Project2PixelPlane(Pc2);

        const Eigen::Vector2d r = px2 - obvs[j];
        double chi2 = r.squaredNorm();
        Eigen::Vector2d rho;
        HuberLoss(chi2, rho);

        const double d = 1 / Pc2.z();
        const double d2 = 1. / pow(Pc2.z(), 2);
        J_Pc2Norm_Pc2 << d, 0, -Pc2.x() * d2, 0, d, -Pc2.y() * d2, 0, 0, 0;
        const Eigen::Matrix<double, 2, 3> J_px2_Pc2 =
            J_px2_Pc2Norm * J_Pc2Norm_Pc2 * rho[1];

        const Eigen::Vector3d dt = Pw1 - Twc2.t_wb_;
        // * Pc2 w.r.t R12
        J_Pc2_Twc2.block<3, 3>(0, 0) = SkewSymmetric(Tc2w.q_wb_ * dt);

        // 给整体雅可比矩阵赋值
        // H = J'*J, g = -J'*b;
        // 当前雅可比及梯度的行和列，用于构建上述H矩阵和g向量
        const int ai = j * resDim, aj = poseStartCol;
        const int bi = j * resDim, bj = pointStartCol + j * kPointDim;
        const Eigen::Matrix<double, 2, 6> A = J_px2_Pc2 * J_Pc2_Twc2;
        double w = 1.0;  // 1.0 / lk1s[i]->invDepthCov_;
        H.block<6, 6>(aj, aj) += (A.transpose() * A) * w;

        /******** -J.T * b的size为[J.cols() x 1]**************
            * | A.T  C.T  E.T |       | A.T*b1 + C.T*b2 + E.T*b3|
            * | B.T  D.T  F.T | * b = | B.T*b1 + D.T*b2 + F.T*b3|
            *
        *****************************************************/
        g.middleRows(aj, 6) -= A.transpose() * r * w;
    }
}

Optimizer::ResidualInfo Optimizer::ResampleStablePwAndObvCurFrame(
    const double sampleRatio, const Pose& Twc2,
    vector<Eigen::Vector3d>& stablePws, vector<Eigen::Vector2d>& stableObvs) {
    const Camera& cam = *cam_;
    const Pose Tc2w = Twc2.Inverse();

    vector<pair<int, double>> idx2Chi2(stablePws.size());
    for (size_t j = 0; j < stablePws.size(); ++j) {
        const Eigen::Vector3d pc = Tc2w * stablePws[j];
        const Eigen::Vector2d px = cam.Project2PixelPlane(pc);

        // 必须与计算Jacobian的残差计算方式一致
        // 注意： cost = p.T * p = [1x1]向量，我们是对cost进行线性化，因此求导的对象是r^2，
        // 而胡伯核函数的自变量是r^2
        idx2Chi2[j] = pair(j, (px - stableObvs[j]).squaredNorm());
    }
    sort(idx2Chi2.begin(), idx2Chi2.end(),
         [](const pair<int, double>& p1, pair<int, double>& p2) {
             return p1.second < p2.second;
         });
    const size_t maxUsefulId =
        static_cast<size_t>(idx2Chi2.size() * sampleRatio);

    vector<Eigen::Vector3d> samplePws;
    vector<Eigen::Vector2d> sampleObvs;
    samplePws.reserve(stablePws.size());
    sampleObvs.reserve(stablePws.size());
    ResidualInfo info;
    for (size_t i = 0; i < maxUsefulId; ++i) {
        const int id = idx2Chi2[i].first;
        samplePws.emplace_back(stablePws[id]);
        sampleObvs.emplace_back(stableObvs[id]);
        Eigen::Vector2d rho;  // 残差值和核函数关于残差的导数
        HuberLoss(idx2Chi2[i].second, rho);
        info.cost += rho[0];
        ++info.totalConstraintNum;
        ++info.usefulLandmarkNum;
    }

    info.meanCost = info.cost / info.totalConstraintNum;
    stablePws.swap(samplePws);
    stableObvs.swap(sampleObvs);

    cout << fmt::format(
        "stable Pws size:{}, resample ratio: {:.3f}, max resample id: {}, max "
        "resample chi2: {:.1f}, resample num: {}, constraint num: {}, mean "
        "reproj cost: {:.1f}.\n",
        samplePws.size(), sampleRatio, maxUsefulId,
        idx2Chi2[maxUsefulId].second, stablePws.size(), info.totalConstraintNum,
        info.meanCost);
    return info;
}

#if USE_SPARSE_H_MATRIX
Eigen::VectorXd Optimizer::SchurCompleteSolve(
    const Eigen::SparseMatrix<double>& H, const Eigen::VectorXd& b,
    const int poseNum, const int pointNum, const bool firstTime,
    const bool logOut) {
    chrono::steady_clock::time_point tStart = chrono::steady_clock::now();

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
    const int poseSize = poseNum * kPoseDim;
    const int pointSize = H.cols() - poseSize;

    // 使用稀疏矩阵视图，避免拷贝数据
    const auto& A = H.block(0, 0, poseSize, poseSize);
    const auto& D = H.block(poseSize, poseSize, pointSize, pointSize);

    chrono::steady_clock::time_point t0 = chrono::steady_clock::now();
    std::vector<Eigen::Triplet<double>> DinvMatTriplets;
    if (firstTime) {
        DinvMatTriplets.reserve(pointNum);
    }
    for (int i = 0; i < pointSize; ++i) {
        //Dinv.block(i, i, kPointDim, kPointDim).noalias() = D.block(i, i, kPointDim, kPointDim).inverse();
        if (D.coeff(i, i) > 1e-12 || D.coeff(i, i) < -1e-12) {
            Dinv_[i] = 1.0 / D.coeff(i, i);
        } else {
            Dinv_[i] = 0.;
        }
    }

    const auto& B = H.block(0, poseSize, poseSize, pointSize);
    const auto& C = H.block(poseSize, 0, pointSize, poseSize);

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    //const Eigen::MatrixXd E = -B * Dinv;
    // E的计算耗时最长，利用Dinv是稀疏矩阵这一特性加速
#if 0
    for (int i = 0; i < B.rows(); ++i) {
        for (int j = 0; j < B.cols(); ++j)
            E_(i, j) = -B.coeff(i, j) * Dinv_[j];
    }
#else
    // 利用B矩阵的稀疏性进行遍历
    E_.setZero();
    int* m_outerStarts = H_.outerIndexPtr();
    int* m_innerRowId = H_.innerIndexPtr();
    double* m_Values = H_.valuePtr();

    // #pragma omp parallel for schedule(static)
    for (int j = poseSize; j < H_.outerSize(); ++j) {
        // 第col列的非零元素范围
        // m_outerStarts记录到j这一列时，一共消耗了多少innerIndex数量，即有多少个非0元素
        const int start = m_outerStarts[j];
        const int end = m_outerStarts[j + 1];
        const double s = -Dinv_[j - poseSize];
        // 遍历当前第j列的innerIndex行索引
        for (int k = start; k < end; ++k) {
            const int rowId = m_innerRowId[k];
            if (rowId < poseSize) {
                const double value = m_Values[k];
                // E_是[poseSize x pointSize]维度矩阵，
                // B也是[poseSize x pointSize]，注意填充位置变化
                E_(rowId, j - poseSize) = value * s;
            } else {
                break;
            }
        }
    }
#endif
    // E_.noalias() = -B;
    // for (int j = 0; j < E_.cols(); ++j) {
    //     E_.col(j) *= Dinv_(j, j);
    // }
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();

    Eigen::VectorXd deltaX = Eigen::VectorXd::Zero(poseSize + pointSize);
    // 求pose增量
    newA_ = A + E_ * C;
    chrono::steady_clock::time_point t3 = chrono::steady_clock::now();
    //cout << "newA:\n" << newA << endl;
    // 根据leftMatrix矩阵的稀疏性，这里不需要其完整形式即可计算出new_b
    //Eigen::VectorXd new_b = leftMatrix * b;
    // | I  E|
    // | 0  I| * b
    Eigen::VectorXd new_b = b;
    new_b.head(poseSize) = b.head(poseSize) + E_ * b.tail(pointSize);
    chrono::steady_clock::time_point t4 = chrono::steady_clock::now();

    deltaX.head(poseSize) =
        newA_.colPivHouseholderQr().solve(new_b.head(poseSize));
    chrono::steady_clock::time_point t5 = chrono::steady_clock::now();
    //cout << "b: " << b.transpose() << endl
    //     << "newb: " << new_b.transpose() << endl;
    // 求point增量
    // H * Δx = b ==> C*deltaX_pose + D*deltaX_point = b
    // D*deltaX_point = b - C*deltaX_pose
    // deltaX_point = D.inv * (b - C*deltaX_pose)
    // 由于Dinv_是稀疏的对角线矩阵，避免不必要的加法，需注意使用array
    // Eigen::VectorXd deltaPoint =
    //     Dinv_ * (new_b.tail(pointSize) - C * deltaPose);
    deltaX.tail(pointSize) =
        Dinv_.cwiseProduct(new_b.tail(pointSize) - C * deltaX.head(poseSize));
    chrono::steady_clock::time_point t6 = chrono::steady_clock::now();

    if (logOut) {
        cout << fmt::format(
            "cal D.inv spend: {:.1f}ms, "
            "cal E mat: {:.1f}ms, "
            "cal newA mat: {:.1f}ms, "
            "cal new_b vec: {:.1f}ms, "
            "cal dPose: {:.1f}ms, "
            "cal dPoint: {:.1f}ms, "
            "total spend: {:.1f}ms\n",
            ChronoMillisecTimeDuration(t0, t1),
            ChronoMillisecTimeDuration(t1, t2),
            ChronoMillisecTimeDuration(t2, t3),
            ChronoMillisecTimeDuration(t3, t4),
            ChronoMillisecTimeDuration(t4, t5),
            ChronoMillisecTimeDuration(t5, t6),
            ChronoMillisecTimeDuration(tStart, t6));
    }

    return deltaX;
}

void Optimizer::ConstructSparseMatrixMapTable() {
    colMajorSparseMatrixRowId2DataPtr_.clear();
    colMajorSparseMatrixRowId2DataPtr_.resize(
        H_.cols(), vector<double*>(H_.cols(), nullptr));
    sparseHmatrixElementNum_ = H_.nonZeros();
    cout << "sparse Hessian matrix noZeros num: " << sparseHmatrixElementNum_
         << endl;
    // Eigen SparseMatrix示例：3×3 矩阵
    // [ 1.0  0.0  4.0 ]
    // [ 0.0  3.0  5.0 ]
    // [ 2.0  0.0  6.0 ]

    // 元素位置索引
    // m_values:      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    // 每个数字都对应着对应列所在的行数，如1.0在第0行，3.0在第1行，6.0在第3行，
    // 它的个数与value数组的元素个数一致
    // m_innerIndices:[0,    2,   1,   0,   1,   2]
    // 第0列索引从0开始，第1列从2开始，那么第0列有2个元素
    // 6为哨兵，记录了数据个数
    // m_outerStarts: [0,    2,   3,   6]  // 第i列从m_values[outerStarts[i]]开始,
    double* m_values = H_.valuePtr();
    int* m_outerStarts = H_.outerIndexPtr();
    int* m_innerRowId = H_.innerIndexPtr();
    for (int j = 0; j < H_.cols(); ++j) {
        int startValueId = m_outerStarts[j];
        int endValueId = m_outerStarts[j + 1];
        auto& colMarjor2RowDataPtr = colMajorSparseMatrixRowId2DataPtr_[j];
        for (int i = startValueId; i < endValueId; ++i) {
            // 直接存入H_(m_innerRowId[i], j)元素的指针
            // colMarjor2RowDataPtr.insert({m_innerRowId[i], &m_values[i]});
            colMarjor2RowDataPtr[m_innerRowId[i]] = &m_values[i];
        }
    }
}

#else
Eigen::VectorXd Optimizer::SchurCompleteSolve(const Eigen::MatrixXd& H,
                                              const Eigen::VectorXd& b,
                                              const int poseNum,
                                              const int pointNum,
                                              const bool& logOut) {
    chrono::steady_clock::time_point tStart = chrono::steady_clock::now();

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
    const int poseSize = poseNum * kPoseDim;
    const int pointSize = H.cols() - poseSize;

    // 这里若使用Eigen::MatrixXd&，那么会产生临时对象，导致内存分配，对于2000个地图点可能需要10ms完成，浪费巨大！！！
    const Eigen::Block<const Eigen::MatrixXd>& A =
        H.block(0, 0, poseSize, poseSize);
    const Eigen::Block<const Eigen::MatrixXd>& D =
        H.block(poseSize, poseSize, pointSize, pointSize);

    chrono::steady_clock::time_point t0 = chrono::steady_clock::now();
    // 只需在分配内存时置0即可，难点是信息矩阵的重置，因为H_矩阵是使用+=
    // SetEigenMatrixAll0(Dinv_);
    chrono::steady_clock::time_point tAssian = chrono::steady_clock::now();

#pragma omp parallel for
    for (int i = 0; i < pointSize; i += kPointDim) {
        //Dinv.block(i, i, kPointDim, kPointDim).noalias() = D.block(i, i, kPointDim, kPointDim).inverse();
        if (D(i, i) > 1e-12 || D(i, i) < 1e-12) {
            Dinv_[i] = 1.0 / D(i, i);
        } else {
            Dinv_[i] = 0.;  // 由于只reset一次
        }
    }
    Eigen::VectorXd deltaX = Eigen::VectorXd::Zero(poseSize + pointSize);
    if (A.isApproxToConstant(0)) {
        // 仅更新point
        deltaX.tail(pointSize) = Dinv_.cwiseProduct(b.tail(pointSize));
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
    const Eigen::Block<const Eigen::MatrixXd>& B =
        H.block(0, poseSize, poseSize, pointSize);
    const Eigen::Block<const Eigen::MatrixXd>& C =
        H.block(poseSize, 0, pointSize, poseSize);
    // cout << "B - C.T:\n" << B-C.transpose() <<endl;

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    //const Eigen::MatrixXd E = -B * Dinv;
    // E的计算耗时最长，利用Dinv是稀疏矩阵这一特性加速
    // SetEigenMatrixAll0(E_);
    for (int i = 0; i < B.rows(); ++i) {
        for (int j = 0; j < B.cols(); ++j)
            E_(i, j) = -B(i, j) * Dinv_[j];
    }
    // E_.noalias() = -B;
    // for (int j = 0; j < E_.cols(); ++j) {
    //     E_.col(j) *= Dinv_(j, j);
    // }
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();

    // 这是个稀疏矩阵，可以优化掉
    //Eigen::MatrixXd leftMatrix(H.rows(), H.cols());
    //leftMatrix.setIdentity();
    //leftMatrix.block(0, 0, poseSize, poseSize).setIdentity();
    //leftMatrix.block(0, poseSize, poseSize, pointSize).noalias() = E;
    //leftMatrix.block(poseSize, 0, pointSize, poseSize).setZero();
    //leftMatrix.block(poseSize, poseSize, pointSize, pointSize).setIdentity();

    // 求pose增量
    newA_.noalias() = A + E_ * C;
    //cout << "newA:\n" << newA << endl;
    // 根据leftMatrix矩阵的稀疏性，这里不需要其完整形式即可计算出new_b
    //Eigen::VectorXd new_b = leftMatrix * b;
    // | I  E|
    // | 0  I| * b
    Eigen::VectorXd new_b = b;
    new_b.head(poseSize) = b.head(poseSize) + E_ * b.tail(pointSize);
    Eigen::VectorXd deltaPose = newA_.llt().solve(new_b.head(poseSize));
    chrono::steady_clock::time_point t3 = chrono::steady_clock::now();
    //cout << "b: " << b.transpose() << endl
    //     << "newb: " << new_b.transpose() << endl;
    // 求point增量
    // H * Δx = b ==> C*deltaX_pose + D*deltaX_point = b
    // D*deltaX_point = b - C*deltaX_pose
    // deltaX_point = D.inv * (b - C*deltaX_pose)
    // 由于Dinv_是稀疏的对角线矩阵，避免不必要的加法，需注意使用array
    // Eigen::VectorXd deltaPoint =
    //     Dinv_ * (new_b.tail(pointSize) - C * deltaPose);
    Eigen::VectorXd deltaPoint =
        Dinv_.cwiseProduct(new_b.tail(pointSize) - C * deltaPose);
    chrono::steady_clock::time_point t4 = chrono::steady_clock::now();

    // cout << setprecision(5) << "deltaPoint: "<< deltaPoint.transpose() << endl;
    deltaX.head(poseSize) = deltaPose;
    deltaX.tail(pointSize) = deltaPoint;
    chrono::steady_clock::time_point t5 = chrono::steady_clock::now();

    if (logOut) {
        cout << fmt::format(
            "reference memory spend: {:.1f}ms, assign memory spend: {:.1f}, "
            "calculate D.inv spend: {:.1f}ms, "
            "calculate E mat spend: {:.1f}ms, "
            "calculate dPose spend: {:.1f}ms, "
            "calculate dPoint spend: {:.1f}ms, construct dX spend: {:.1f}ms, "
            "total spend: {:.1f}\n",
            ChronoMillisecTimeDuration(tStart, t0),
            ChronoMillisecTimeDuration(t0, tAssian),
            ChronoMillisecTimeDuration(tAssian, t1),
            ChronoMillisecTimeDuration(t1, t2),
            ChronoMillisecTimeDuration(t2, t3),
            ChronoMillisecTimeDuration(t3, t4),
            ChronoMillisecTimeDuration(t4, t5),
            ChronoMillisecTimeDuration(tStart, t5));
    }

    return deltaX;
}
#endif

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
    chrono::steady_clock::time_point time1 = chrono::steady_clock::now();
    int continousNoImprovementNum = 0;
    bool acceptNewVariableStatus = true;
    WinBApreAssignMatrixMemory();
    int lastAcceptBAupdateTime = 0;
    double lambda = config->initLambda;
    bool converge = false;
    int inerIte = -1;
    for (int acceptIte = 0; acceptIte < config->maxIterationLM;) {
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        ++inerIte;
        if (acceptNewVariableStatus) {
            // 只在状态量更新时需要重新计算信息矩阵和梯度，以节省大量的计算时间
            ConstructJ_H_b_g(inerIte == 0);
            lastAcceptBAupdateTime = acceptIte;
        }

        if (inerIte == 0) {
            AdaptSetInitLambda(H_, lambda);
            cout << "window BA: H_.diag: " << H_.diagonal().head(12).transpose()
                 << "\ng_: " << g_.head(12).transpose() << "\n";
        }

        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();

        if (margKFstatus_) {
#if !USE_SPARSE_H_MATRIX
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
#endif
        } else {
            // 破坏了优化问题一致性，不可取
            //_lambda.head(6).setConstant(
            //    g_.head(6).cwiseAbs().maxCoeff() * 1e4);  // 首帧的约束足够大，但不能使矩阵病态
            //_lambda[0] = 1e20;
            if (inerIte == 0) {
                cout << "Fixed First Frame!!! lambda: " << lambda << endl;
            }
        }
        for (int i = 0; i < H_.rows(); ++i) {
#if USE_SPARSE_H_MATRIX
            *(colMajorSparseMatrixRowId2DataPtr_[i][i]) += lambda;
#else
            H_(i, i) += lambda;
#endif
        }

        Eigen::VectorXd delta_x;

        chrono::steady_clock::time_point t3 = chrono::steady_clock::now();
#if USE_SPARSE_H_MATRIX
        delta_x = SchurCompleteSolve(H_, g_, window_.size(),
                                     lastCost.usefulLandmarkNum, inerIte == 0,
                                     inerIte % 20 == 0);
#else
        delta_x =
            SchurCompleteSolve(H_, g_, window_.size(),
                               lastCost.usefulLandmarkNum, inerIte % 20 == 0);
#endif
        chrono::steady_clock::time_point t4 = chrono::steady_clock::now();

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
        chrono::steady_clock::time_point t5 = chrono::steady_clock::now();
        UpdateStatusVariables(delta_x);
        chrono::steady_clock::time_point t6 = chrono::steady_clock::now();
        // 判断当前更新是否有效，在使用新的pose计算cost时，可能会让一些点被设置为noUsed
        ResidualInfo newCost = CalculateResidualWindow();
        chrono::steady_clock::time_point t7 = chrono::steady_clock::now();

        if (margKFstatus_) {
            // 需要考虑先验残差约束
            newCost.priorConstraintChi2 = CalculatePriorCost(delta_x);
            newCost.cost += newCost.priorConstraintChi2;
        }

        // 使用LM方法，考虑存在由于图像模糊投影不上的问题，因此newCost不能小于0
        double predictReduction = 1.0;
        if (newCost.cost < lastCost.cost) {
            predictReduction =
                ComputePredictionReduction(lambda, delta_x, g_, H_);
        }
        UpdateLMlambda(lastCost, newCost, predictReduction,
                       acceptNewVariableStatus, continousNoImprovementNum,
                       lambda);
        if (!acceptNewVariableStatus) {
            for (size_t i = 0; i < optLandmark_.size(); ++i) {
                if (!optLandmark_[i]->NoUsed()) {
                    optLandmark_[i]->BackUpStatus();
                }
            }
            for (size_t i = 0; i < window_.size(); ++i) {
                window_[i]->BackUpStatus();
            }
        } else {
            converge = LMstopJudge(continousNoImprovementNum, lambda, lastCost,
                                   newCost, delta_x);
            lastCost = newCost;
            // 更新先验残差构成信息项
            if (margKFstatus_) {
                UpdatePriorDeltaX0(delta_x);
            }
            ++acceptIte;  // 一次迭代成功
        }

        if (acceptNewVariableStatus) {
            chrono::steady_clock::time_point t8 = chrono::steady_clock::now();
            cout << fmt::format(
                "useful lk num: {}, acceptIte: {}, inerIte: {}, "
                "ConstructJ_H_b_g spend: {:.1f}ms, Add "
                "lambda: {:.1f}, "
                "SchurCompleteSolve: {:.1f}ms, Copy status: "
                "{:.3f}ms, UpdateStatusVariables: {:.1f}ms, "
                "CalculateResidualWindow: {:.1f}ms, "
                "ComputePredictionReduction and Update: {:.1f}ms, LM "
                "one iteration: {:.1f}ms\n",
                newCost.usefulLandmarkNum, acceptIte, inerIte,
                ChronoMillisecTimeDuration(t1, t2),
                ChronoMillisecTimeDuration(t2, t3),
                ChronoMillisecTimeDuration(t3, t4),
                ChronoMillisecTimeDuration(t4, t5),
                ChronoMillisecTimeDuration(t5, t6),
                ChronoMillisecTimeDuration(t6, t7),
                ChronoMillisecTimeDuration(t7, t8),
                ChronoMillisecTimeDuration(t1, t8));
        }

        if (converge || lambda > config->maxLambdaValueLM) {
            break;
        }
    }
    // 同时更新地图点备份状态
    for (size_t i = 0; i < optLandmark_.size(); ++i) {
        if (!optLandmark_[i]->NoUsed() && optLandmark_[i]->initialized_) {
            optLandmark_[i]->CopyStatus();
        }
    }
    for (size_t i = 0; i < window_.size(); ++i) {
        window_[i]->CopyStatus();
    }
    chrono::steady_clock::time_point time2 = chrono::steady_clock::now();
    const double spendTime = ChronoMillisecTimeDuration(time1, time2);
    cout << fmt::format(
        "First cost: {:.1f}, final cost: {:.1f}, first mean proj cost: {:.1f}, "
        "last mean proj cost: {:.1f}, gradient norm: {:.3f}, "
        "priorConstraintChi2: {:.1f}, usefulLandmark num: {}, "
        "cost decrease ratio: {:.1f}%, usefulNum ratio: {:.1f}%, total "
        "optimize "
        "spend: {:.3f}ms in window, lastAcceptBAupdateTime: {}\n",
        firstCost.cost, lastCost.cost, firstCost.meanCost, lastCost.meanCost,
        g_.norm(), lastCost.priorConstraintChi2, lastCost.usefulLandmarkNum,
        ((firstCost.cost - lastCost.cost) / firstCost.cost) * 100,
        double(lastCost.usefulLandmarkNum) / optLandmark_.size() * 100,
        spendTime, lastAcceptBAupdateTime);

    return lastCost.cost < firstCost.cost;
}

bool Optimizer::ExecuteWindowOptimizeCeres() {
    ResidualInfo lastCost = SetOptimizeStatusVariableForWindowBA();

    // 丢失追踪，重新进行
    if (lastCost.usefulLandmarkNum < 10) {
        cerr << fmt::format("lastCost.usefulLandmarkNum: {} too small!!!\n",
                            lastCost.usefulLandmarkNum);
        return false;
    }

    // 构建优化问题
    ceres::Problem problem;
    // 1、先指定需要参与优化的参数块对象
    ceres::Manifold* se3Parameterization = new SE3Parameterization;
    // qw, qx, qy, qz, qw
    unordered_map<KeyFrame*, array<double, 7>> vecTwc;
    for (KeyFrame* kf : window_) {
        const Eigen::Quaterniond& q = kf->Twc_.q_wb_;
        const Eigen::Vector3d& p = kf->Twc_.t_wb_;
        vecTwc.insert({kf, {q.w(), q.x(), q.y(), q.z(), p.x(), p.y(), p.z()}});
        problem.AddParameterBlock(vecTwc[kf].data(), 7, se3Parameterization);
    }
    problem.SetParameterBlockConstant(vecTwc[window_[0]].data());
    const bool canFixSecondKF =
        window_.size() >
        static_cast<size_t>(config->maxKFnumInWindow / 2.0 + 0.5);
    if (canFixSecondKF) {
        problem.SetParameterBlockConstant(vecTwc[window_[1]].data());
    }

    vector<double> vecInvZ1(optLandmark_.size());
    ceres::LossFunction* huberLoss = new ceres::HuberLoss(config->huberDelta);
    for (size_t i = 0; i < optLandmark_.size(); ++i) {
        auto p = optLandmark_[i];
        vecInvZ1[i] = p->invZ_;
        if (p->NoUsed()) {
            continue;
        }

        // 注意，必须确保每个landmark都能构建残差以成为状态变量，否则优化变量位置会有问题，导致最终更新出错
        KeyFrame* host = p->host_;
        bool depthParameterAdd = false;
        for (const auto& kf2obv : p->target_) {
            // 需要注意每个关键帧、每个landmark在H矩阵中的位置
            KeyFrame* target = kf2obv.first;
            if (target == host) {
                continue;
            }
            const auto& p2 = target->kpts_.row(kf2obv.second);
            ceres::CostFunction* costFunction = new ProjectInvDepthResidual(
                p->GetHostFrameObv(), Eigen::Vector2d(p2[0], p2[1]),
                cam_->K_[0]);
            if (!depthParameterAdd) {
                depthParameterAdd = true;
                problem.AddParameterBlock(&vecInvZ1[i], 1);
            }
            problem.AddResidualBlock(costFunction, huberLoss,
                                     vecTwc[host].data(), vecTwc[target].data(),
                                     &vecInvZ1[i]);
        }
    }

    // 配置优化选项
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 50;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    //options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    options.preconditioner_type = ceres::SCHUR_JACOBI;

    options.minimizer_type = ceres::TRUST_REGION;
    options.trust_region_strategy_type = ceres::DOGLEG;

    // options.function_tolerance = 1e-12;
    // options.gradient_tolerance = 1e-16;
    // options.parameter_tolerance = 1e-12;

    options.use_explicit_schur_complement = true;

    options.num_threads = 4;

    options.max_solver_time_in_seconds = 0.5;

    //options.logging_type =
    //    ceres::PER_MINIMIZER_ITERATION;  // 设置输出log便于bug排查

    /*
    // Openvins 配置
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.preconditioner_type = ceres::SCHUR_JACOBI;
    options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.linear_solver_ordering = ordering;
    options.function_tolerance = 1e-5;
    options.gradient_tolerance = 1e-4 * options.function_tolerance;
    // 禁止调用glog
    options.logging_type = ceres::LoggingType::SILENT;
*/
    // 运行优化
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // 输出结果
    std::cout << summary.BriefReport() << std::endl;

    for (auto& kf2Pose : vecTwc) {
        const auto& d = kf2Pose.second;
        const Eigen::Quaterniond q(d[0], d[1], d[2], d[3]);
        const Eigen::Vector3d p(d[4], d[5], d[6]);
        kf2Pose.first->SetTwc(Pose(q, p));
    }

    for (size_t i = 0; i < optLandmark_.size(); ++i) {
        optLandmark_[i]->SetInvZvalue(vecInvZ1[i]);
    }

    return true;
}

void Optimizer::PreSelectLandmarkForTracking(vector<shared_ptr<Landmark>>& lk1s,
                                             vector<Eigen::Vector2d>& obvs) {
    lk1s.clear();
    obvs.clear();
    lk1s.reserve(globalOptFlw.trackLandmark_.size());
    obvs.reserve(lk1s.size());
    constexpr int kDebugNum = 20000;

    auto SelectLandmark = [&lk1s, &obvs](
                              const vector<shared_ptr<Landmark>>& trackLandmark,
                              const vector<cv::Point2f>& prevPts) {
        for (size_t i = 0; i < trackLandmark.size(); ++i) {
            // TODO: FEJ指的是关于逆深度的线性化点在首次计算出逆深度值时
            shared_ptr<Landmark> lk = trackLandmark[i];
            if (lk->initialized_ && !lk->CanBeDelete()) {
                lk1s.emplace_back(lk);
                const cv::Point2f& p = prevPts[i];
                obvs.emplace_back(p.x, p.y);
#if defined(WRITE_MATCH_PAIR_IMAGE)
                //DrawProjectCase(*lk, obvs.back(), globalOptFlw.prevImg_, Twc2);
#endif
            }
            if (lk1s.size() > kDebugNum) {
                break;
            }
        }
    };

    {
        //lock_guard<mutex> lock(globalOptFlwMutex); // 调用处加锁
        SelectLandmark(globalOptFlw.trackLandmark_, globalOptFlw.prevPts_);

        cout << fmt::format(
            "curF pose opt preselect landmark num: {}, total optflow track "
            "feature "
            "num: {}\n",
            lk1s.size(), globalOptFlw.trackLandmark_.size());
    }
}

bool Optimizer::OptimizeCurFrame(Pose& Twc2, const int curFid,
                                 int& totalPointNum, int& usefulPointNum,
                                 ResidualInfo& info) {

    // 构建优化问题所需观测
    vector<shared_ptr<Landmark>> preLks;
    vector<Eigen::Vector2d> preObvs;
    vector<Eigen::Vector3d> stablePws;
    vector<Eigen::Vector2d> stableObvs;
    ResidualInfo lastCost;
    {
        lock_guard<mutex> lock(globalOptFlwMutex);
        PreSelectLandmarkForTracking(preLks, preObvs);
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
        // 此处globalOptFlw只访问图像的size，不修改其内容，不需要加锁
        // 注意：前端只能访问globalOptFlw，不能修改它
        lastCost = SetOptimizeLandmarkForTracking(
            preLks, preObvs, Twc2, globalOptFlw.prevImg_, usefulPointNum,
            stablePws, stableObvs);
    }

    if (stablePws.size() < 100) {
        cout << "use " << stablePws.size() << " landmarks to optimize!\n";
    }

    ResidualInfo firstCost = lastCost;
    if (firstCost.usefulLandmarkNum < 20) {
        cout << fmt::format("Error first useful constrint num: {}\n",
                            firstCost.usefulLandmarkNum);
        return false;
    }

    double lambda = config->initLambda;
    const vector<double> iterativeUsefulResidualRatio{0.95};
    for (size_t ite = 0; ite <= iterativeUsefulResidualRatio.size(); ++ite) {
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

        if (ite > 0) {
            const ResidualInfo temp = ResampleStablePwAndObvCurFrame(
                iterativeUsefulResidualRatio[ite - 1], Twc2, stablePws,
                stableObvs);
            if (stablePws.size() < 50) {
                info = lastCost;
                return sqrt(info.meanCost) <
                       config->maxMeanProjectResidual2CreateKF;
            }
            firstCost = lastCost = temp;
            lambda = 100.0;
        }

        // 执行迭代优化
        int continousNoImprovementNum = 0;
        bool acceptNewVariableStatus = true;
        Eigen::Matrix<double, 6, 6> hessianMatrix;
        Eigen::Matrix<double, 6, 1> gradient;
        constexpr int kMaxIterativeTime = 100;
        bool converge = false;
        int inerIte = -1;
        for (int acceptIte = 0; acceptIte < kMaxIterativeTime;) {
            ++inerIte;
            if (acceptNewVariableStatus) {
                // 只需要在状态量更新的时候重新线性化一次即可，以节省计算时间
                CalculateHandGradiantCurFrame(stablePws, stableObvs, Twc2,
                                              hessianMatrix, gradient);
            }
            if (inerIte == 0) {
                cout << "frame BA: hessianMatrix.diag: "
                     << hessianMatrix.diagonal().transpose()
                     << "\ng: " << gradient.transpose() << "\n";
                AdaptSetInitLambda(hessianMatrix, lambda);
            }

            // debug, 返回J, 判断H, g计算的正确性
            hessianMatrix.diagonal().array() += lambda;
            const Eigen::Matrix<double, 6, 1> delta_x =
                hessianMatrix.ldlt().solve(gradient);

            const Pose poseBackup = Twc2;

            // 当前帧pose状态更新
            Twc2.Update(delta_x.middleRows(0, 3), delta_x.middleRows(3, 3));

            // 判断当前更新是否有效
            ResidualInfo newCost =
                CalculateResidualCurFrame(stablePws, stableObvs, Twc2);

            if (config->iterateLogFreqLM > 0 &&
                acceptIte % config->iterateLogFreqLM == 0) {
                cout << setprecision(5) << "delta_pose: " << delta_x.transpose()
                     << ", norm: " << delta_x.norm() << "\n";
                cout << fmt::format(
                    "CurF BA iterate {} times, firstCost: {:.1f}, "
                    "lastCost: "
                    "{:.1f}, newCost: "
                    "{:.1f}, "
                    "useful landmark: {}, "
                    "lambda: {}.\n",
                    acceptIte, firstCost.cost, lastCost.cost, newCost.cost,
                    newCost.usefulLandmarkNum, lambda);
            }

            double predictReduction = 1.0;
            if (newCost.cost < lastCost.cost) {
                predictReduction = ComputePredictionReductionFrame(
                    lambda, delta_x, gradient, hessianMatrix);
            }
            UpdateLMlambda(lastCost, newCost, predictReduction,
                           acceptNewVariableStatus, continousNoImprovementNum,
                           lambda);
            // LM 方法
            if (!acceptNewVariableStatus) {
                Twc2 = poseBackup;
            } else {
                converge = LMstopJudge(continousNoImprovementNum, lambda,
                                       lastCost, newCost, delta_x);
                lastCost = newCost;
                ++acceptIte;  // 更新成功才算一次迭代
            }

            if (converge || lambda > config->maxLambdaValueLM) {
                cout << fmt::format(
                    "frame BA iterative info: ite: {}, stop i: {}\n", ite,
                    acceptIte);
                break;
            }
        }

        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        const double spendTime = ChronoMillisecTimeDuration(t1, t2);
        cout << fmt::format(
            "First cost: {:.1f}, final cost: {:.1f}, first mean proj cost: "
            "{:.1f}, "
            "last mean proj cost: {:.1f}, \n"
            "cost decrease ratio: {:.1f}%, usefulNum ratio: {:.1f}%, total "
            "optimize "
            "spend: {:.3f}ms.\n",
            firstCost.cost, lastCost.cost, firstCost.meanCost,
            lastCost.meanCost,
            ((firstCost.cost - lastCost.cost) / firstCost.cost) * 100,
            double(lastCost.usefulLandmarkNum) / stablePws.size() * 100,
            spendTime);

        info = lastCost;
    }

    return sqrt(info.meanCost) < config->maxMeanProjectResidual2CreateKF;
}

bool Optimizer::OptimizeCurFrameCeres(Pose& Twc2, const int curFid,
                                      int& totalPointNum, int& usefulPointNum,
                                      ResidualInfo& info) {
    // 构建优化问题所需观测
    vector<shared_ptr<Landmark>> preLks;
    vector<Eigen::Vector2d> preObvs;
    vector<Eigen::Vector3d> stablePws;
    vector<Eigen::Vector2d> stableObvs;
    ResidualInfo lastCost;
    {
        lock_guard<mutex> lock(globalOptFlwMutex);
        PreSelectLandmarkForTracking(preLks, preObvs);
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
        // 此处globalOptFlw只访问图像的size，不修改其内容，不需要加锁
        // 注意：前端只能访问globalOptFlw，不能修改它
        lastCost = SetOptimizeLandmarkForTracking(
            preLks, preObvs, Twc2, globalOptFlw.prevImg_, usefulPointNum,
            stablePws, stableObvs);
    }

    if (stablePws.size() < 100) {
        cout << "use " << stablePws.size() << " landmarks to optimize!\n";
    }

    if (lastCost.usefulLandmarkNum < 20) {
        cout << fmt::format("Error first useful constrint num: {}\n",
                            lastCost.usefulLandmarkNum);
        return false;
    }

    const auto& q = Twc2.q_wb_;
    const auto& p = Twc2.t_wb_;
    array<double, 7> optTwc2{q.w(), q.x(), q.y(), q.z(), p.x(), p.y(), p.z()};
    ceres::Problem problem;
    // 1、先指定需要参与优化的参数块对象
    ceres::Manifold* se3Parameterization = new SE3Parameterization;
    problem.AddParameterBlock(optTwc2.data(), 7, se3Parameterization);

    ceres::LossFunction* huberLoss = new ceres::HuberLoss(config->huberDelta);
    for (size_t i = 0; i < stablePws.size(); ++i) {
        const Eigen::Vector3d& pw = stablePws[i];
        const Eigen::Vector2d& obv = stableObvs[i];
        ceres::CostFunction* costFunc =
            new ProjectionResidual(pw, obv, cam_->K_[0]);
        problem.AddResidualBlock(costFunc, huberLoss, optTwc2.data());
    }

    // 2. 优化器配置
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 100;
    options.linear_solver_type = ceres::DENSE_SCHUR;

    options.minimizer_type = ceres::TRUST_REGION;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;

    // options.function_tolerance = 1e-12;
    // options.gradient_tolerance = 1e-16;
    // options.parameter_tolerance = 1e-12;

    // 3. 运行优化
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // 输出结果
    std::cout << "Frame BA: " << summary.BriefReport() << std::endl;

    const Eigen::Quaterniond q2(optTwc2[0], optTwc2[1], optTwc2[2], optTwc2[3]);
    const Eigen::Vector3d p2(optTwc2[4], optTwc2[5], optTwc2[6]);
    Twc2.q_wb_ = q2;
    Twc2.t_wb_ = p2;

    info = SetOptimizeLandmarkForTracking(preLks, preObvs, Twc2,
                                          globalOptFlw.prevImg_, usefulPointNum,
                                          stablePws, stableObvs);

    return true;
}

void Optimizer::AddOneKeyFeame(KeyFrame* kf) {
    if (window_.empty()) {
        const int totalTrackLandmarkNum = kf->InitializeLandmark(nullptr);
        cout << fmt::format(
            "First kf id: {}, optical flow total feature num: {}\n", kf->id_,
            totalTrackLandmarkNum);
        window_.emplace_back(kf);
    } else {
        lock_guard<mutex> lock(newKFmutex_);
        // newKFqueue_.push(kf);

        // TODO：不再这里三角化
        TriangulateNewLandmark(kf);
        newKF_ = kf;
    }
}

void Optimizer::TriangulateNewLandmark(KeyFrame* kf) {
    // 直接在新KF中提取关键点，并放入optical flow结构中，同时保留上一KF的跟踪结果仍进行跟踪
    KeyFrame* lastKf = window_.back();
    const int totalTrackLandmarkNum = kf->InitializeLandmark(lastKf);
    cout << fmt::format("kf id: {}, optical flow total feature num: {}\n",
                        kf->id_, totalTrackLandmarkNum);
    int historyTriSucceedNum = 0;
    int prevTriSucceedNum = 0;
    int failTriNum = 0;

    unordered_map<KeyFrame*, Pose> kf2T12;
    {
        lock_guard<mutex> lock(globalOptFlwMutex);
        for (size_t i = 0; i < globalOptFlw.trackLandmark_.size(); ++i) {
            auto lk = globalOptFlw.trackLandmark_[i];
            if (lk == nullptr || lk->initialized_) {
                continue;
            }

            // 仍被当前帧观测到，可以进行深度滤波更新，或者进行多视角优化
            const cv::Point2f& p = globalOptFlw.prevPts_[i];  // curFrameObv
            const Eigen::Vector2d curObv(p.x, p.y);

            // 三角化
            double idepth1 = 0, idepth2 = 0;
            if (!kf2T12.count(lk->host_)) {
                kf2T12[lk->host_] = lk->host_->Tcw_ * kf->Twc_;
            }
            const Pose& T12 = kf2T12.at(lk->host_);
            if (!GetHostAndCurFrameObservationDepth(lk->GetHostFrameObv(),
                                                    curObv, cam_->Kinv_[0], T12,
                                                    idepth1, idepth2)) {
                ++lk->failInitializeNum_;
                ++failTriNum;
                continue;
            }
            lk->SetTriangulateResult(idepth1);
            if (i < globalOptFlw.historyLandmarkNum_) {
                ++historyTriSucceedNum;
            } else {
                ++prevTriSucceedNum;
            }

#if defined(WRITE_MATCH_PAIR_IMAGE)
//DrawTriangulateCase(idepth1, *lk, curObv.cast<int>(),
//                    kf->debugGrayImg_, T12);
#endif
        }

        int removeFeatNum = window_.back()->RemoveNoInitializeLongFeature();
        cout << fmt::format(
            "Triangulate by KF_{} report: prevTriSucceedNum: {}, "
            "historyTriSucceedNum: {}, remove long time "
            "fail initialize feature num: {}, failTriNum: {}.\n",
            kf->id_, prevTriSucceedNum, historyTriSucceedNum, removeFeatNum,
            failTriNum);
    }

#if defined(WRITE_MATCH_PAIR_IMAGE)
    WriteDebugTriangulateCase2Video(kf->id_);
#endif

    // CalculateLastKFmeanDepth();
    cout << fmt::format("add kf id: {}, kf time duration: {:.3f}s\n", kf->id_,
                        kf->timestamp_ - window_.back()->timestamp_);
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

    int transformLandmarkNum = 0;
    int newAddOptimizeLandmarkNum = 0;
    // 保留superpoint点集的深度值，用于闭环sim3计算
    oldest->depth_.resize(oldest->desc_.rows());
    oldest->depth_.setConstant(-1.0);
    for (int i = 0; i < oldest->kpts_.rows(); ++i) {
        auto lk = oldest->landmark_[i];
        if (lk == nullptr) {
            continue;
        }

        if (lk->host_ != oldest) {
            if (i < oldest->desc_.rows() && lk->initialized_ &&
                !lk->CanBeDelete()) {
                const Eigen::Vector3d pc = oldest->Tcw_ * lk->GetPw();
                if (pc.z() > config->minDepth && pc.z() < config->maxDepth) {
                    oldest->depth_[i] = pc.z();
                }
            }

            continue;  // 不需要转换
        } else if (i < oldest->desc_.rows() && lk->invZ_ > 0 &&
                   lk->initialized_ && !lk->CanBeDelete()) {
            const double z = 1.0 / lk->invZ_;
            if (z > config->minDepth && z < config->maxDepth) {
                oldest->depth_[i] = z;
            }
        }
        bool transformSucceed = false;

        // 把地图点的所有权转移到最新KF，其余的不要
        // 这里需要将lk从oldestKF中删除，并将其添加到下一个KF，且需要保持地址不变
        // if (lk->TransformHost2OtherKF(nextKF)) {
        if (lk->TransformHost2NextKeyframe(window_)) {
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

        for (auto lk : kf->landmark_) {
            if (lk == nullptr) {
                // TODO：需要完美处理SetCanDelete的情况，这里只是跳过了被置为不合法的情况，
                // 优化Landmark*的管理
                continue;
            }
            if (lk->target_.count(oldest)) {
                lk->target_.erase(oldest);
            }
        }
    }
    // 移除光流跟踪中被标记为可以删除的Landmark
    auto RemoveDeleteLandmarkFromOpticalFlow =
        [](vector<std::shared_ptr<Landmark>>& lks,
           vector<cv::Point2f>& obvs) -> void {
        vector<std::shared_ptr<Landmark>>::iterator it1 = lks.begin();
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

    {
        lock_guard<mutex> lock(globalOptFlwMutex);
        RemoveDeleteLandmarkFromOpticalFlow(globalOptFlw.trackLandmark_,
                                            globalOptFlw.prevPts_);
    }

    // 有可能闭环在更新pose
    if (margKFid != config->maxKFnumInWindow) {
        lock_guard<mutex> lock(vecMargKfMutex_);  // 更新耗时可忽略
        // 当前实现只会遍历前几帧，push_back不影响，但要预申请内存，避免扩容导致异常
        vecMargKf_.emplace_back(oldest);
        cout << "vecMargKf_ size: " << vecMargKf_.size() << endl;
    } else {
        // 删除老帧看看是否会有影响
        delayEraseKeyframe_.push_back(oldest);
    }

    if (delayEraseKeyframe_.size() > 1) {
        lock_guard<mutex> lock(KeyFrame::mutexForSyncView3Dstatus);
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
    const int minObvNum = window_.size() > 2 ? kMinUsefulObvNumWithHost : 2;

    for (int i = startKFid; i < static_cast<int>(window_.size()); ++i) {
        for (auto p : window_[i]->landmark_) {
            // 地图点有被其他关键帧看到
            if (p == nullptr || !p->initialized_ || p->IsOutOfRange() ||
                p->CanBeDelete() ||
                static_cast<int>(p->target_.size()) < minObvNum) {
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
        auto& p = optLandmark_[i];
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
            const KeyFrame* target = kf2obv.first;
            // 这里如果还未被标记为noUsed_，那么会多计算几个残差，
            // 但是影响不大，可能会出现
            if (target == host) {
                continue;
            }

            const Eigen::Vector3d pc2 =
                useBackUpStatus ? target->TcwBack_ * pw : target->Tcw_ * pw;
            const Eigen::Vector2d px2 = cam_->Project2PixelPlane(pc2);

            Eigen::Vector2d r = px2 - target->GetObv(kf2obv.second);
            double chi2 = r.squaredNorm();
            Eigen::Vector2d rho;
            HuberLoss(chi2, rho);
            info.cost += rho[0];
            ++info.totalConstraintNum;
        }
    }

    // cout << fmt::format(
    //     "window BA: residual info.all.cost: {:.1f}, useful landmark num: {}, "
    //     "total constraint num: {}, mean cost: {:.1f}\n",
    //     info.cost, info.usefulLandmarkNum, info.totalConstraintNum,
    //     info.meanCost);

    if (isnan(info.cost) || isinf(info.cost) || info.totalConstraintNum == 0) {
        cout << fmt::format("Error window cost value: {}, landmark num: {}\n",
                            info.cost, info.usefulLandmarkNum);
        info.cost = DBL_MAX;
    } else {
        info.meanCost = info.cost / info.totalConstraintNum;
    }
    return info;
}

void Optimizer::DebugOptlandmarkStatus(const size_t num, const string& name) {
    string debugUsefulLkIndex(name + " useful lk status:[ ");
    for (size_t i = 0; i < min(optLandmark_.size(), num); ++i) {
        auto p = optLandmark_[i];
        debugUsefulLkIndex.append(fmt::format(
            " {}-{}-{}", i, reinterpret_cast<size_t>(p.get()), p->NoUsed()));
    }
    debugUsefulLkIndex.append(" ]");
    cout << debugUsefulLkIndex << endl;
}

Optimizer::ResidualInfo Optimizer::SetOptimizeStatusVariableForWindowBA(
    KeyFrame* const margKF) {

    vector<double> sumErrorVec(optLandmark_.size(), 0.);
    vector<int> sumConstraintNum(optLandmark_.size(), 0);
    int totalSelectConstraintNum = 0;
    for (size_t i = 0; i < optLandmark_.size(); ++i) {
        auto& p = optLandmark_[i];
        KeyFrame* host = p->host_;
        if (host->id_ == window_.back()->id_) {
            sumErrorVec[i] += kMaxSetError;
            continue;
        }

        ResidualInfo tempInfo;
        const Eigen::Vector3d pc1 = p->GetPc();
        const Eigen::Vector3d pw = host->Twc_ * pc1;
        for (const auto& kf2obv : p->target_) {
            const KeyFrame* target = kf2obv.first;
            // 这里如果还未被标记为noUsed_，那么会多计算几个残差，
            // 但是影响不大，可能会出现
            if (target == host) {
                continue;
            }

            const Eigen::Vector3d pc2 = target->Tcw_ * pw;
            const Eigen::Vector2d px2 = cam_->Project2PixelPlane(pc2);
            const bool inRange = InRange(target->grayImg_, px2.cast<int>());
            if (!inRange || pc2.z() < kMinSceneDepthInCamera) {
                sumErrorVec[i] += kMaxSetError;
                break;
            }

            Eigen::Vector2d r = px2 - target->GetObv(kf2obv.second);
            tempInfo.cost = max(tempInfo.cost, r.squaredNorm());
            ++tempInfo.totalConstraintNum;
            ++totalSelectConstraintNum;
        }

        const int excludeNum =
            (margKF != nullptr && p->target_.count(margKF)) ? 1 : 0;
        const int usefulConstraintNum =
            window_.size() == 2 ? 1 : (kMinUsefulObvNum + excludeNum);
        if (tempInfo.totalConstraintNum >= usefulConstraintNum) {
            sumErrorVec[i] += tempInfo.cost;
            sumConstraintNum[i] += tempInfo.totalConstraintNum;
        } else {
            sumErrorVec[i] += kMaxSetError;
        }
    }

    vector<double> errorCopy = sumErrorVec;
    sort(sumErrorVec.begin(), sumErrorVec.end());

    int usefulIdx = sumErrorVec.size() - 1;
    while (usefulIdx >= 0 && sumErrorVec[usefulIdx] >= kMaxSetError) {
        --usefulIdx;
    }
    const double maxChi2 =
        max(config->maxProjectError * config->maxProjectError,
            sumErrorVec[static_cast<int>(kMaxErrorRatio * usefulIdx)]);
    cout << fmt::format(
        "win BA error range: [{:.1f}, {:.1f}], usefulIdx: {}, maxChi2: "
        "{:.1f}, "
        "sumErrorVec.size: {}\n",
        sumErrorVec.front(), sumErrorVec.back(), usefulIdx, maxChi2,
        sumErrorVec.size());

    ResidualInfo info;
    for (size_t i = 0; i < optLandmark_.size(); ++i) {
        auto& p = optLandmark_[i];
        if (errorCopy[i] > maxChi2) {
            p->SetNoUsed();
            continue;
        }

        const double& chi2 = errorCopy[i];
        Eigen::Vector2d rho;
        HuberLoss(chi2, rho);
        info.cost += errorCopy[i];
        info.totalConstraintNum += sumConstraintNum[i];
        ++info.usefulLandmarkNum;
    }

    cout << fmt::format(
        "window BA: totalSelectConstraintNum: {}, set opt variable "
        "residual "
        "info.all.cost: {:.1f}, useful "
        "landmark num: {}, "
        "total constraint num: {}, mean cost: {:.1f}\n",
        totalSelectConstraintNum, info.cost, info.usefulLandmarkNum,
        info.totalConstraintNum, info.meanCost);

    if (isnan(info.cost) || isinf(info.cost) || info.totalConstraintNum == 0) {
        cout << fmt::format("Error window cost value: {}, landmark num: {}\n",
                            info.cost, info.usefulLandmarkNum);
        info.cost = DBL_MAX;
    } else {
        info.meanCost = info.cost / info.totalConstraintNum;
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
    //const int margDim = kPoseDim + margLandmark.size() * depthDim;
    // 这里我们直接将最老帧的landmark转移或丢弃不用，只保留边缘化最老帧的信息
    const int margDim = kPoseDim;
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
#if USE_SPARSE_H_MATRIX
    // 稀疏矩阵模式下，暂不用实现边缘化
    const Eigen::VectorXd deltaX;
#else
    const Eigen::VectorXd deltaX = SchurCompleteSolve(
        Hp_, g_p_, config->maxKFnumInWindow,
        Hp_.cols() - config->maxKFnumInWindow * config->maxKFnumInWindow,
        false);
#endif
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

#if USE_SPARSE_H_MATRIX
double Optimizer::ComputePredictionReduction(
    const double lambda, const Eigen::VectorXd& deltaX,
    const Eigen::VectorXd& g, const Eigen::SparseMatrix<double>& H) {
    // 实际下降值为： lastCost - newCost
    // g = -J.T * r
    return -0.5 * deltaX.dot(H * deltaX) + deltaX.dot(g) -
           0.5 * lambda * deltaX.squaredNorm();
}
#else
double Optimizer::ComputePredictionReduction(const double lambda,
                                             const Eigen::VectorXd& deltaX,
                                             const Eigen::VectorXd& g,
                                             const Eigen::MatrixXd& H) {
    // 实际下降值为： lastCost - newCost
    // g = -J.T * r
    return -0.5 * deltaX.dot(H * deltaX) + deltaX.dot(g) -
           0.5 * lambda * deltaX.squaredNorm();
}
#endif

double Optimizer::ComputePredictionReductionFrame(
    const double lambda, const Eigen::Matrix<double, 6, 1>& deltaX,
    const Eigen::Matrix<double, 6, 1>& g,
    const Eigen::Matrix<double, 6, 6>& H) {
    // 实际下降值为： lastCost - newCost
    // g = -J.T * r
    return -0.5 * deltaX.dot(H * deltaX) + deltaX.dot(g) -
           0.5 * lambda * deltaX.squaredNorm();
}

void Optimizer::UpdateLMlambda(const Optimizer::ResidualInfo& lastCost,
                               const Optimizer::ResidualInfo& newCost,
                               const double predictReduction, bool& accept,
                               int& continousNoImprovementNum, double& lambda) {
    const double costRelativeAbsDiff = lastCost.cost - newCost.cost;
    const double rho = costRelativeAbsDiff / (predictReduction + 1e-12);
    if (rho > 0) {
        if (rho > 0.75) {
            lambda = max(0.3 * lambda, 1e-9);
        } else {
            lambda = max(lambda, 1e-9);
        }
        accept = true;
        continousNoImprovementNum = 0;
    } else {
        lambda *= 1.5 * (1 + 0.1 * continousNoImprovementNum);
        accept = false;
        ++continousNoImprovementNum;
    }
}

bool Optimizer::LMstopJudge(const int& continousNoImprovementNum,
                            const double lambda, const ResidualInfo& last,
                            const ResidualInfo& cur,
                            const Eigen::VectorXd& delta) {
    const double costRelativeAbsDiff =
        last.cost - cur.cost;  // cur一定要比last小
    if (costRelativeAbsDiff < config->convergeCostDiffLM * last.cost) {
        cout << fmt::format("LM cost diff: {} converge!\n",
                            costRelativeAbsDiff);
        return true;
    }
    if (lambda > config->maxLambdaValueLM) {
        cout << fmt::format("lambad too large: {}\n", lambda);
        return true;
    }
    if (delta.norm() < config->convergeMaxDeltaXValueLM) {
        return true;
    }

    return false;
}

void Optimizer::ConstructJ_H_b_g(const bool firstTime) {
    // 构建H, g
    // 给出每个KF对应的在H矩阵中的位置
    unordered_map<const KeyFrame*, int> kfMapCol;
    unordered_map<const KeyFrame*, int> debugKFMapResidualNum;
    unordered_map<const KeyFrame*, Eigen::Matrix3d> Rcw;
    unordered_map<const KeyFrame*, Eigen::Matrix3d> Rwc;
    for (size_t i = 0; i < window_.size(); ++i) {
        kfMapCol.insert({window_[i], i * 6});
        debugKFMapResidualNum.insert(
            {window_[i], 0});  // 统计每个图像对应的residual数量
        Rcw.insert({window_[i], window_[i]->Tcw_.q_wb_.toRotationMatrix()});
        Rwc.insert({window_[i], window_[i]->Twc_.q_wb_.toRotationMatrix()});
    }

    // 或许我们不知道residual，Jacobian的行数，但是H矩阵以及g向量的维度是可知的

    const bool canFixSecondKF =
        window_.size() >
        static_cast<size_t>(config->maxKFnumInWindow / 2.0 + 0.5);

    // SetEigenMatrixAll0(H_, logOut);
    g_.setZero();
#if USE_SPARSE_H_MATRIX
    std::vector<Eigen::Triplet<double>> triplets;
    if (firstTime) {
        triplets.resize(window_.size() * kPoseDim * kPoseDim +
                        optLandmark_.size() * kPointDim);
    } else {
        // H_.setZero();  // 稀疏矩阵置0，会修改内存结构，需要使用memset
        // 将H_矩阵置0且不修改其内部内存结构
        memset(H_.valuePtr(), 0., sizeof(double) * sparseHmatrixElementNum_);
    }
#endif
    // clang-format off
    // 计算residual & jacobian
    /*********
    *    T0 T1 ... d0 d1 ...
    * r0
    *********/
    // clang-format on
    const int depthStartCol = window_.size() * kPoseDim;
    const int resDim = 2;
    int resNum = 0;  // 显示当前计算到雅可比的第几行
    int usefulLandmarkNum = 0;
    // res w.r.t (u2, v2) [2x2]的单位矩阵
    // px2 w.r.t Pc2 [2x3]
    Eigen::Matrix<double, 2, 3> J_px2_Pc2Norm = cam_->K_[0].block(0, 0, 2, 3);
    J_px2_Pc2Norm(0, 2) = 0.;
    J_px2_Pc2Norm(1, 2) = 0.;

    // Pc2 w.r.t Twc2 : Pc2 = Twc2.inv * Pw
    Eigen::Matrix<double, 3, 6> J_Pc2_Twc2;  // ---> optimization variable
    Eigen::Matrix<double, 3, 6> J_Pw_Twc1;   // ---> optimization variable
    // Pw w.r.t Pwc1
    J_Pw_Twc1.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
    // Pc1 w.r.t z
    Eigen::Vector3d J_Pc1_z(0, 0, 0);  // --------> optimization variable

    // Residual w.r.t optimization variables Jacobian
    Eigen::Matrix<double, 2, 6> A1, A2;
    Eigen::Matrix<double, 6, 2> A1t, A2t;
    Eigen::Matrix<double, 2, 1> B;
    Eigen::Matrix<double, 1, 2> Bt;

    hasResetHessianblock_.clear();
    for (size_t i = 0; i < optLandmark_.size(); ++i) {
        auto p = optLandmark_[i];
        if (p->NoUsed()) {
            continue;
        }

        // 注意，必须确保每个landmark都能构建残差以成为状态变量，否则优化变量位置会有问题，导致最终更新出错
        KeyFrame* host = p->host_;
        const Eigen::Vector3d pc1 = p->GetPc();
        const Eigen::Vector3d pw = host->Twc_ * pc1;

        // 最新关键帧没有反向追踪能力
        bool addConstraint = false;
        const Eigen::Vector3d pc1Norm(p->GetPcNorm());

        for (const auto& kf2obv : p->target_) {
            // 需要注意每个关键帧、每个landmark在H矩阵中的位置
            const KeyFrame* target = kf2obv.first;
            if (target == host) {
                continue;
            }

            const Eigen::Vector3d pc2 = target->Tcw_ * pw;
            const Eigen::Vector2d px2 = cam_->Project2PixelPlane(pc2);

            // 经过校验，可以构建residual和jacobian
            const Eigen::Vector2d r = px2 - target->GetObv(kf2obv.second);

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
            const Eigen::Matrix<double, 2, 3> J_res_Pc2 = J_px2_Pc2 * rho[1];

            const Eigen::Vector3d dt = pw - target->Twc_.t_wb_;
            // Pc2 w.r.t Rwc2
            J_Pc2_Twc2.block<3, 3>(0, 0) =
                SkewSymmetric(target->Tcw_.q_wb_ * dt);
            // Pc2 w.r.t Pwc2
            J_Pc2_Twc2.block<3, 3>(0, 3) = -Rcw[target];

            // Pc2 w.r.t Pw
            const Eigen::Matrix3d& J_Pc2_Pw = Rcw[target];

            // Pw w.r.t Twc1 : Pw = Twc1 * Pc1 = Rwc1 * pc1 + Pwc1
            // Pw w.r.t Rwc1
            J_Pw_Twc1.block<3, 3>(0, 0).noalias() =
                -Rwc[host] * SkewSymmetric(pc1);

            // Pw w.r.t Pc1
            const Eigen::Matrix3d& J_Pw_Pc1 = Rwc[host];

            // Pc1 w.r.t z
            // 使用逆深度表示
            const double d1 = pow(p->invZ_, 2);
            J_Pc1_z << -pc1Norm.x() / d1, -pc1Norm.y() / d1, -1 / d1;

            const Eigen::Matrix<double, 3, 1> J_Pw_z = J_Pw_Pc1 * J_Pc1_z;

            // Residual w.r.t optimization variables Jacobian
            if (!margKFstatus_) {
                if (host == window_[0] || host == window_[1]) {
                    // fixed滑动窗口第一帧
                    A1.setZero();
                } else {
                    A1.noalias() = J_res_Pc2 * J_Pc2_Pw * J_Pw_Twc1;
                }

                if (canFixSecondKF && target == window_[1]) {
                    A2.setZero();
                } else {
                    A2.noalias() = J_res_Pc2 * J_Pc2_Twc2;
                }
            }

            B.noalias() = J_res_Pc2 * J_Pc2_Pw * J_Pw_z;

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
            A1t = A1.transpose();
            A2t = A2.transpose();
            Bt = B.transpose();

#if USE_SPARSE_H_MATRIX
            if (firstTime) {
                EmplaceBackTriplet<6, 6>(a1j, a1j, A1t * A1 * w, triplets);
                EmplaceBackTriplet<6, 6>(a1j, a2j, A1t * A2 * w, triplets);
                EmplaceBackTriplet<6, 1>(a1j, bj, A1t * B * w, triplets);

                EmplaceBackTriplet<6, 6>(a2j, a1j, A2t * A1 * w, triplets);
                EmplaceBackTriplet<6, 6>(a2j, a2j, A2t * A2 * w, triplets);
                EmplaceBackTriplet<6, 1>(a2j, bj, A2t * B * w, triplets);

                EmplaceBackTriplet<1, 6>(bj, a1j, Bt * A1 * w, triplets);
                EmplaceBackTriplet<1, 6>(bj, a2j, Bt * A2 * w, triplets);
                // EmplaceBackTriplet(bj, bj, Bt * Bt * w, triplets_);
                triplets.emplace_back(bj, bj, Bt * B);
            } else {
                // 直接向sparse matrix H_赋值
                auto& data = colMajorSparseMatrixRowId2DataPtr_;
                UpdateSparseHessianMatrix<6, 6>(a1j, a1j, A1t * A1 * w, data);
                UpdateSparseHessianMatrix<6, 6>(a1j, a2j, A1t * A2 * w, data);
                UpdateSparseHessianMatrix<6, 1>(a1j, bj, A1t * B * w, data);

                UpdateSparseHessianMatrix<6, 6>(a2j, a1j, A2t * A1 * w, data);
                UpdateSparseHessianMatrix<6, 6>(a2j, a2j, A2t * A2 * w, data);
                UpdateSparseHessianMatrix<6, 1>(a2j, bj, A2t * B * w, data);

                UpdateSparseHessianMatrix<1, 6>(bj, a1j, Bt * A1 * w, data);
                UpdateSparseHessianMatrix<1, 6>(bj, a2j, Bt * A2 * w, data);
                // H_.coeffRef(bj, bj) += Bt * B;
                *data[bj][bj] += Bt * B;
            }
#else
            MatrixBlockReset<6, 6>(
                a1j, a1j,
                reinterpret_cast<uint64_t>(&H_.block<6, 6>(a1j, a1j)(0, 0)));
            H_.block<6, 6>(a1j, a1j) += (A1t * A1) * w;

            MatrixBlockReset<6, 6>(
                a1j, a2j,
                reinterpret_cast<uint64_t>(&H_.block<6, 6>(a1j, a2j)(0, 0)));
            H_.block<6, 6>(a1j, a2j) += A1t * A2 * w;

            MatrixBlockReset<6, 1>(
                a1j, bj,
                reinterpret_cast<uint64_t>(&H_.block<6, 1>(a1j, bj)(0, 0)));
            H_.block<6, 1>(a1j, bj) += A1t * B * w;

            MatrixBlockReset<6, 6>(
                a2j, a1j,
                reinterpret_cast<uint64_t>(&H_.block<6, 6>(a2j, a1j)(0, 0)));
            H_.block<6, 6>(a2j, a1j) += A2t * A1 * w;

            MatrixBlockReset<6, 6>(
                a2j, a2j,
                reinterpret_cast<uint64_t>(&H_.block<6, 6>(a2j, a2j)(0, 0)));
            H_.block<6, 6>(a2j, a2j) += (A2t * A2) * w;

            MatrixBlockReset<6, 1>(
                a2j, bj,
                reinterpret_cast<uint64_t>(&H_.block<6, 1>(a2j, bj)(0, 0)));
            H_.block<6, 1>(a2j, bj) += A2t * B * w;

            MatrixBlockReset<1, 6>(
                bj, a1j,
                reinterpret_cast<uint64_t>(&H_.block<1, 6>(bj, a1j)(0, 0)));
            H_.block<1, 6>(bj, a1j) += Bt * A1 * w;

            MatrixBlockReset<1, 6>(
                bj, a2j,
                reinterpret_cast<uint64_t>(&H_.block<1, 6>(bj, a2j)(0, 0)));
            H_.block<1, 6>(bj, a2j) += Bt * A2 * w;

            MatrixBlockReset<1, 1>(
                bj, bj,
                reinterpret_cast<uint64_t>(&H_.block<1, 1>(bj, bj)(0, 0)));
            H_.block<1, 1>(bj, bj) += (B.transpose() * B) * w;
#endif
            // clang-format off
            /********************* 利用稀疏性计算g=-J'*b ****************************
            * | A1'|       | A1' * b |
            * | A2'| * b = | A2' * b |
            * | B' |       | B'  * b |
            **********************************************************************/
            // clang-format on
            g_.middleRows(a1j, kPoseDim) -= A1t * r * w;
            g_.middleRows(a2j, kPoseDim) -= A2t * r * w;
            g_.middleRows(bj, kPointDim) -= Bt * r * w;

            addConstraint = true;
        }

        if (addConstraint) {
            ++usefulLandmarkNum;
        }
    }

#if USE_SPARSE_H_MATRIX
    if (firstTime) {
        H_.setFromTriplets(triplets.begin(), triplets.end());
        H_.makeCompressed();
        ConstructSparseMatrixMapTable();
    }
#endif
}

void Optimizer::ConstructRelativePoseConstraint(Eigen::MatrixXd& H,
                                                Eigen::VectorXd& g) {
    if (config->relativePoseConstraintWeight <= 0.) {
        return;
    }
    const int resDim = 6;

    H.resize(window_.size() * kPoseDim, window_.size() * kPoseDim);
    g.resize(window_.size() * kPoseDim);
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

        const int aj1 = (i - 1) * kPoseDim, aj2 = i * kPoseDim;
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
    if (window_.size() < 2) {
        optLandmark_.clear();
        return false;
    }

    cout << fmt::format(
                "Begin SlidingWindowOptimize! margKFid: {}, Sample landmark "
                "num for "
                "window BA: {}.",
                margKFid, sampleNum)
         << endl;

    //if (margKFid >= 0 && margKFid < config->maxKFnumInWindow) {
    //    // 将待删除的最老帧移到滑窗开头，有可能移除最新帧
    //    MoveMargKF2FirstPosInWindow(margKFid);
    //    margKFstatus_ = false;
    //    if (TransformLandmarkOwnerFromOldestKF(margKFid)) {
    //        // 只需要保留最老帧的信息即可，或者只固定首帧的pose进行优化在debug阶段也是可取的
    //        // 其信息已经通过深度点的传播转移到后面的KF中
    //        if (margKFid < 2 && config->useMarginalization) {
    //            margKFstatus_ = MarginalizeOldestKeyFrame();
    //            cout << fmt::format("marg kf succeed: {}\n", margKFstatus_);
    //        }

    //        // 如果是使用点-点匹配逻辑的话，那么应该先进行边缘化再转移点的控制权
    //        // 产生的问题是：那些没有host被边缘化，但是没有target的点不造成影响
    //        // 那些host被边缘化，但是仍有target的点，可能只剩一个target本身的观测
    //        RemoveOldestKeyFrame(margKFid);
    //    }
    //}

    chrono::steady_clock::time_point t0 = chrono::steady_clock::now();
#if USE_CERES2
    const bool winOptSuccess = ExecuteWindowOptimizeCeres();
#else
    const bool winOptSuccess = ExecuteWindowOptimize();
#endif
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    const double spendTime = ChronoMillisecTimeDuration(t0, t1);
    cout << fmt::format("win size: {}, win BA spend {:.3f}ms!\n",
                        window_.size(), spendTime);
    if (margKFid == config->maxKFnumInWindow || 1) {
        // BA之后移除最新帧
        MoveMargKF2FirstPosInWindow(margKFid);
        TransformLandmarkOwnerFromOldestKF(margKFid);
        RemoveOldestKeyFrame(margKFid);

        // 意味着最新KF被接受，由于闭环优化耗时长，这里不等待
        if (lastTryLoopNewKf_ == nullptr &&
            margKFid != config->maxKFnumInWindow &&
            lastTryLoopNewKfMutex_.try_lock()) {
            lastTryLoopNewKf_ = window_.back();
            lastTryLoopNewKfMutex_.unlock();
        }
    }
    const int markDeleteNum = MarkBigResidualLandmarkDelete();
    cout << fmt::format("markDeleteNum: {}, winOptSuccess: {}\n", markDeleteNum,
                        winOptSuccess);

    return winOptSuccess;
}

void Optimizer::HuberLoss(const double chi2, Eigen::Vector2d& rho) {

    //const double scale = 1.0 / pow(2, lvl);
    const double& huberDelta = config->huberDelta;  //  * scale;
    const double& huberDelta2 = config->huberDelta2;
    const double sqrtChi2 = sqrt(chi2);

    if (chi2 > huberDelta2) {
        if (sqrtChi2 < huberDelta) {
            cerr << fmt::format("sqrtChi2: {} < huberDelta: {}\n", sqrtChi2,
                                huberDelta);
        }
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

    int historyTrackInitLandmarkNum = 0;
    for (const auto& lk : globalOptFlw.trackLandmark_) {

        historyTrackInitLandmarkNum += static_cast<int>(
            lk->initialized_ &&
            lk->host_ != window_[config->maxKFnumInWindow - 1]);
    }
    cout << fmt::format(
                "Select marg kf historyTrackInitLandmarkNum: {}, "
                "historyTrackRatio: {:.2f}",
                historyTrackInitLandmarkNum,
                globalOptFlw.GetHistoryTrackFeatureRatio())
         << endl;
    // 历史地图点足够多，并且历史跟踪特征点足够多时，才把最新帧用作三角化
    bool canRemoveNewestKf = historyTrackInitLandmarkNum > 200 &&
                             globalOptFlw.GetHistoryTrackFeatureRatio() > 0.7;
    //if (historyTrackInitLandmarkNum > 200 &&
    //    globalOptFlw.GetHistoryTrackFeatureRatio() > 0.5) {
    //    // 直接移除最新帧，但会导致BA优化无效
    //    return window_.size();
    //}

    int smallId = 0;
    size_t minTrackFeatureNum = 1e10;
    // 说明预设关键帧数量较少，直接移除最老帧
    if (window_.size() > 3) {
#if 0

        const Eigen::Vector3d posDiff =
            curKF.Twc_.t_wb_ - window_[window_.size() - 2]->Twc_.t_wb_;
        const double horDist = posDiff.head(2).norm();
        cout << fmt::format("curKF hor dist to the last 2 frame: {:.2f}\n",
                            horDist);
        if (horDist < config->needNewKFtrans * 1.5) {
            smallId = window_.size() - 1;  // 移除掉最新的，因为重复观测可能大
            cout << "Will remove the last keyframe from window!\n";
        }
#else
        // 计算每个关键帧被当前最新关键帧观测到的特征点数量，
        // 找出光流特征最少的关键帧，由于光流特性，说明其已经超出当前视野，直接移除该帧
        constexpr size_t kKeepLastKFnum = 1;
        const size_t keepMaxFrameId = window_.size() - kKeepLastKFnum;
        unordered_map<const KeyFrame*, size_t> kf2Index;
        vector<size_t> trackFeatNumEachKF(keepMaxFrameId, 0);
        for (size_t i = 0; i < keepMaxFrameId; ++i) {
            kf2Index.insert({window_[i], i});
        }

        cout << fmt::format(
            "globalOptFlw.prevPts_.size(): {}, "
            "globalOptFlw.historyLandmarkNum_: {}\n",
            globalOptFlw.prevPts_.size(), globalOptFlw.historyLandmarkNum_);

        // 关键帧被当前KF观测到的特征点数量统计
        for (size_t i = 0; i < globalOptFlw.historyLandmarkNum_; ++i) {
            auto lk = globalOptFlw.trackLandmark_[i];
            if (!kf2Index.count(lk->host_)) {
                continue;
            }
            const int kfIdx = kf2Index[lk->host_];
            ++trackFeatNumEachKF[kfIdx];
        }
#endif
        // 找出最小值
        for (size_t i = 0; i < trackFeatNumEachKF.size(); ++i) {
            if (trackFeatNumEachKF[i] < minTrackFeatureNum) {
                smallId = i;
                minTrackFeatureNum = trackFeatNumEachKF[i];
            }
        }

        // 决定移除跟踪最少帧还是上一次添加的新KF
        constexpr int kMinTrackFeatureNum = 200;
        if (smallId != static_cast<int>(window_.size() - 1) &&
            minTrackFeatureNum > kMinTrackFeatureNum) {
            smallId = static_cast<int>(window_.size() - 1);
            cout << fmt::format(
                "will remove the last kf, for minTrackFeatureNum: {} > "
                "{}.\n",
                minTrackFeatureNum, kMinTrackFeatureNum);
        }

        // 打印统计结果
        string logInfo(
            "track feature num of each kf in window by newest kf:\n");
        for (size_t i = 0; i < trackFeatNumEachKF.size(); ++i) {
            if (i == 0) {
                logInfo.append(
                    fmt::format("window[{}]: {} ", i, trackFeatNumEachKF[i]));
            } else {
                logInfo.append(
                    fmt::format(", window[{}]: {}", i, trackFeatNumEachKF[i]));
            }
        }
        logInfo.append(
            fmt::format("; will delete window[{}], minTrackFeatureNum: {}\n",
                        smallId, trackFeatNumEachKF[smallId]));
        cout << logInfo;
    }

    if (canRemoveNewestKf && minTrackFeatureNum > 50) {
        return window_.size();
    }

    return smallId;
}

void Optimizer::MoveMargKF2FirstPosInWindow(const int margId) {
    if (margId < 0) {
        return;
    }
    KeyFrame* oldest = window_[margId];
    window_.erase(window_.begin() + margId);
    window_.insert(window_.begin(), oldest);
    KeyFrame::WritePoseMessage2File(*oldest);
}

bool Optimizer::TrackLocalMap(KeyFrame* kf2, bool& trackLocalMapLow) {
    Pose Twc2 = kf2->Twc_;
    // 仅优化当前帧pose，避免由于其运动模糊影响landmark估计值导致系统崩溃
    // 同时加快计算速度
    int totalPointNum = 0;
    int usefulPointNum = 0;
    ResidualInfo info;
    onlyPoseUpdate_ = true;
#if USE_CERES2
    const bool optSuccess = OptimizeCurFrameCeres(Twc2, kf2->id_, totalPointNum,
                                                  usefulPointNum, info);
#else
    const bool optSuccess =
        OptimizeCurFrame(Twc2, kf2->id_, totalPointNum, usefulPointNum, info);
#endif
    onlyPoseUpdate_ = false;
    const double usefulRatio = double(usefulPointNum) / totalPointNum;
    cout << fmt::format(
        "track totalPointNum: {}, usefulPointNum: {}, usefulRatio: {:.1f}, "
        "mean residual: {}\n",
        totalPointNum, usefulPointNum, usefulRatio, info.meanCost);
    trackLocalMapLow = usefulPointNum < 50;

    Pose beforeTwc2 = kf2->Twc_;
    if (optSuccess) {
        kf2->SetTwc(Twc2);  // 关闭这个出现错乱，证明优化有效
    } else {
        WriteDebugTrackLostStatus(*kf2);
    }
    cout << "cur frame pose diff: " << beforeTwc2.Inverse() * kf2->Twc_ << endl;
    return optSuccess;
}

void Optimizer::AdaptSetInitLambda(const Eigen::MatrixXd& H, double& lambda) {
    // lambda_ = 1.0;
    lambda = max(H.diagonal().maxCoeff() * 1e-4, config->initLambda);
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
    constexpr int kMinObvStableTime = 4;
    constexpr double kMaxChi2 = 9 * 9;
    int markCount = 0;
    for (size_t i = 0; i < optLandmark_.size(); ++i) {
        auto lk = optLandmark_[i];

        KeyFrame* host = lk->host_;
        const Eigen::Vector3d pc1 = lk->GetPc();
        if (pc1.z() < kMinSceneDepthInCamera) {
            lk->SetCanDelete();
            continue;
        }
        const Eigen::Vector3d pw = host->Twc_ * pc1;
        double maxChi2 = 0.;
        for (const auto& kf2obv : lk->target_) {
            const KeyFrame* tar = kf2obv.first;
            if (tar == host) {
                continue;
            }

            if (lk->CanBeDelete()) {
                break;
            }

            const Eigen::Vector3d pc2 = tar->Tcw_ * pw;
            if (pc2.z() < kMinSceneDepthInCamera) {
                lk->SetCanDelete();
                break;
            }
            const Eigen::Vector2d px2 = cam_->Project2PixelPlane(pc2);

            Eigen::Vector2d r = px2 - tar->GetObv(kf2obv.second);
            const double chi2 = r.squaredNorm();
            maxChi2 = chi2 > maxChi2 ? chi2 : maxChi2;
        }

        if (maxChi2 > kMaxChi2 && lk->target_.size() >= kMinObvStableTime) {
            lk->SetCanDelete();
            ++markCount;
        }
    }
    return markCount;
}

void Optimizer::WinBApreAssignMatrixMemory() {
    // 实验发现，执行BA优化时，内存分配操作占总耗时80%以上，真是惊人！！！
    // 此外，不能使用 const Eigen::Matrix& a = A.block()，这样仍会造成临时对象内存分配，
    // 要使用const auto& a = A.block(); 因为block()返回的是 Eigen::Block<const Eigen::MatrixXd>对象

    // 为了避免内存重复分配，LM迭代过程中，不应该再改变状态量维度，若要剔除某个点，直接使其雅可比为0即可，此时该状态量梯度自然变为0
    int optPoseDim = window_.size() * window_[0]->Twc_.Size();
    int landmarkDim = 0;
    for (const auto lk : optLandmark_) {
        landmarkDim += lk->NoUsed() ? 0 : kPointDim;
    }
    const int variableDim = optPoseDim + landmarkDim;
    cout << fmt::format("window_.size: {}, opt variable dim: {}\n",
                        window_.size(), variableDim);

    // 分配求解信息矩阵和梯度的矩阵内存，在SparseMatrix下，会同时将元素置0
    H_.resize(variableDim, variableDim);
    g_.resize(variableDim);

    // 分配舒尔补求解所需矩阵内存
    Dinv_.resize(landmarkDim);
    E_.resize(optPoseDim, landmarkDim);
    newA_.resize(optPoseDim, landmarkDim);

    // 置0舒尔补矩阵，因为其运算是=，只需reset一次
    SetEigenMatrixAll0(E_);
    SetEigenMatrixAll0(newA_);
    // 难点是信息矩阵H_，其运算是+=，采用延迟重置方案
#if !USE_SPARSE_H_MATRIX
    // 稀疏矩阵resize时会同步置0
    SetEigenMatrixAll0(H_, true);
#endif
}

void Optimizer::SetEigenMatrixAll0(Eigen::Matrix<double, -1, -1>& mat,
                                   const bool logOut) {
    // Eigen 大的matrix使用SetZero()函数仍然十分耗时，可能达10ms，这里需要找到一个快速置0的方法
    chrono::steady_clock::time_point t0 = chrono::steady_clock::now();
    memset(mat.data(), 0, mat.size() * sizeof(double));
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    if (logOut) {
        cout << fmt::format("Reset matrix[{}x{}] spend {:.3f}ms\n", mat.rows(),
                            mat.cols(), ChronoMillisecTimeDuration(t0, t1));
    }
}

void Optimizer::RunWindowBA() {
    while (keepRunWindowBA_) {
        // TODO：把三角化移到这里，并取消对于newKF_的判断
        // if (newKFqueue_.empty()) {
        if (newKF_ == nullptr) {
            usleep(2 * 1e3);  // 2ms
            continue;
        }

        chrono::steady_clock::time_point t0 = chrono::steady_clock::now();
        // {
        //     lock_guard<mutex> lock(newKFmutex_);
        //     newKF_ = newKFqueue_.front();
        //     newKFqueue_.pop();
        // }
        // TriangulateNewLandmark();

        // 滑窗优化时，会将当前帧添加到滑窗中去
        SlidingWindowOptimize(newKF_);
        CalculateLastKFmeanDepth();
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        lastWinBAspendTime_ = ChronoMillisecTimeDuration(t0, t1);
        cout << fmt::format(
            "RunWindowBA spend: {:.1f}ms. process newKF_ id: {}, remain "
            "newKFqueue_.size: {}\n",
            lastWinBAspendTime_, newKF_->id_, newKFqueue_.size());

        lock_guard<mutex> lock(newKFmutex_);
        newKF_ = nullptr;
    }
}

void Optimizer::StopRunBA() {
    int tryCount = 0;
    while (newKF_ != nullptr) {
        usleep(10 * 1e3);
        cout << fmt::format("try stop window BA count: {}\n", ++tryCount);
    }
    keepRunWindowBA_ = false;
    cout << fmt::format("Window BA keepRunWindowBA_: {}.\n", keepRunWindowBA_);
}

void Optimizer::RunLoopClosure() {
    while (keepRunLoopClosure_) {
        if (lastTryLoopNewKf_ == nullptr) {
            usleep(10 * 1e3);
            continue;
        }

        int kf1Index = -1;
        {
            // 排序加锁，更新时另外加锁，或者更新放在BA线程后？
            lock_guard<mutex> lock(vecMargKfMutex_);
            sort(vecMargKf_.begin(), vecMargKf_.end(),
                 [](const KeyFrame* a, const KeyFrame* b) {
                     return a->id_ < b->id_;
                 });
            kf1Index = FindLoopClosureKF();
            if (kf1Index < 0) {
                // 是否可能存在死锁？win ba是串行的，应该不会
                lock_guard<mutex> lock(lastTryLoopNewKfMutex_);
                lastTryLoopNewKf_ = nullptr;
                cout << fmt::format(
                            "LP find loop closure kf failed! vecMargKf_ size: "
                            "{}",
                            vecMargKf_.size())
                     << endl;
                continue;
            }
        }

        DynamicPointMatrix Pc1, Pc2;
        if (FindMatchSuperpoint3Dpos(vecMargKf_[kf1Index], lastTryLoopNewKf_,
                                     Pc1, Pc2) < 100) {
            lock_guard<mutex> lock(lastTryLoopNewKfMutex_);
            lastTryLoopNewKf_ = nullptr;
            continue;
        }

        // 解闭环相对位姿约束
        Sim3Pose sT12;
        const double innerRatio = 0.55;
        if (CalculateSim3PosesT12RANSAC(Pc1, Pc2, sT12, 3, 0.999, innerRatio)) {
            cout << "Solve sim3Pose sT12 succeed! sT12:\n" << sT12 << endl;
            // 求解位姿图优化
            vector<KeyFrame*> allKeyframe(vecMargKf_.begin() + kf1Index,
                                          vecMargKf_.end());
            allKeyframe.emplace_back(lastTryLoopNewKf_);
            Sim3PoseGraphOptimizationCeres2(0, allKeyframe.size() - 1, sT12,
                                            allKeyframe);
        } else {
            cout << "Solve sim3Pose sT12 failed!" << endl;
            continue;
        }

        lock_guard<mutex> lock(lastTryLoopNewKfMutex_);
        lastTryLoopNewKf_ = nullptr;
    }
}

int Optimizer::FindLoopClosureKF() {
    // TODO： 使用lightglue寻找，这里暂时使用先验poes实现寻找
    if (vecMargKf_.size() < 50) {
        return -1;
    }

    // 寻找开头5帧，，不能处理大回环内有小回环的情况
    constexpr int kMaxSearchRange = 5;
    constexpr double kMaxLoopClosureDist =
        0.2;  // 尺度漂移时只能由lightglue确定
    for (size_t i = 0; i < kMaxSearchRange; ++i) {
        const KeyFrame* kf = vecMargKf_[i];
        const double posDiff =
            (kf->priorTwc_.Inverse() * lastTryLoopNewKf_->priorTwc_)
                .t_wb_.norm();
        if (posDiff < kMaxLoopClosureDist) {
            // 还是需要lightglue寻找匹配点
            cout << fmt::format(
                        "LP find loop closure kf id: {}, in vecMargKf_ index: "
                        "{}",
                        vecMargKf_[i]->id_, i)
                 << endl;
            return i;
        }
    }

    return -1;
}

int Optimizer::FindMatchSuperpoint3Dpos(const KeyFrame* kf1,
                                        const KeyFrame* kf2,
                                        DynamicPointMatrix& Pc1,
                                        DynamicPointMatrix& Pc2) {
    auto& kpts1 = kf1->kpts_.middleRows(0, kf1->desc_.rows());
    auto& kpts2 = kf2->kpts_.middleRows(0, kf2->desc_.rows());
    Eigen::VectorXf mscores;
    vector<cv::DMatch> matches;
    lightgluePtr->MatchKeypoints(kpts1, kpts2, kf1->desc_, kf2->desc_, mscores,
                                 matches);
    cout << "LP lightglue find 2d match num: " << matches.size() << endl;
    if (matches.size() < 200) {
        return 0;
    }
    int usefulNum = 0;
    for (size_t i = 0; i < matches.size(); ++i) {
        int idx1 = matches[i].queryIdx;
        int idx2 = matches[i].trainIdx;
        const double depth1 = kf1->depth_[idx1];
        const auto lk2 = kf2->landmark_[idx2];
        if (depth1 <= 0 || !lk2 || lk2->CanBeDelete()) {
            continue;
        }
        ++usefulNum;
    }

    Pc1.resize(3, usefulNum);
    Pc2.resize(3, usefulNum);
    const Pose& Tc2w = kf2->Tcw_;
    int idx = 0;
    for (size_t i = 0; i < matches.size(); ++i) {
        int idx1 = matches[i].queryIdx;
        int idx2 = matches[i].trainIdx;
        const double depth1 = kf1->depth_[idx1];
        const auto lk2 = kf2->landmark_[idx2];
        if (depth1 <= 0 || !lk2 || lk2->CanBeDelete()) {
            continue;
        }
        Pc1.col(idx) = cam_->InverseProject(
            {kf1->kpts_.row(idx1)[0], kf1->kpts_.row(idx1)[1]}, depth1);
        Pc2.col(idx) = Tc2w * lk2->GetPw();
        ++idx;
    }
    cout << fmt::format("LP lightglue find useful 3d match num: {} = idx: {}",
                        usefulNum, idx)
         << endl;
    return usefulNum;
}

void Optimizer::StopRunLoopClosure() {
    int tryCount = 0;
    while (lastTryLoopNewKf_ != nullptr) {
        usleep(10 * 1e3);
        cout << fmt::format("try stop loop closure BA count: {}\n", ++tryCount);
    }
    keepRunLoopClosure_ = false;
    cout << fmt::format("Loop closure BA keepRunLoopClosure_: {}.\n",
                        keepRunLoopClosure_);
}

bool Optimizer::Sim3PoseGraphOptimizationCeres2(
    int fixedIndex, int loopClosureIndex, const Sim3Pose& relativeSim3T12,
    vector<KeyFrame*>& allKeyframe) {
    // 构建位姿图
    // 1. 选择闭环内的帧
    vector<KeyFrame*> selectKFresult;
    SelectKeyframeInLoopClosure(allKeyframe, fixedIndex, loopClosureIndex,
                                selectKFresult);
    cout << "Select selectKFresult size: " << selectKFresult.size() << endl;

    // 2. 保留帧间位姿先验
    vector<Sim3Pose> loopClosurePoseTwc;
    vector<Sim3Pose> sT12Constraint;
    CalculateLoopClosureSim3PoseAndConstraint(
        relativeSim3T12, selectKFresult, loopClosurePoseTwc, sT12Constraint);
    ofstream of;
    of.open(kBeforeLoopClosurePoseFilePath);
    for (size_t i = 0; i < loopClosurePoseTwc.size(); ++i) {
        cout << "init sTwc[" << i << "]: " << loopClosurePoseTwc[i].QwbString()
             << ", " << loopClosurePoseTwc[i].PwbString() << endl;
        of << loopClosurePoseTwc[i].DebugOutputPoseMessage() << endl;
    }
    of.close();

    for (size_t i = 0; i < sT12Constraint.size(); ++i) {
        cout << fmt::format("constraint[{}]: {}, {}", i,
                            sT12Constraint[i].QwbString(),
                            sT12Constraint[i].PwbString())
             << endl;
    }
    cout << "Construct sT12Constraint size: " << sT12Constraint.size() << endl;

    ceres::Problem problem;
    // 指定大小，避免内存重分配
    vector<array<double, 8>> vecSim3Pose(loopClosurePoseTwc.size());
    Sim3Parameterization* sim3PoseParameterization = new Sim3Parameterization;
    for (size_t i = 0; i < loopClosurePoseTwc.size(); ++i) {
        const auto& q = loopClosurePoseTwc[i].q_wb_;
        const auto& p = loopClosurePoseTwc[i].t_wb_;
        const double s = loopClosurePoseTwc[i].scale_;
        vecSim3Pose[i] = {q.w(), q.x(), q.y(), q.z(), p.x(), p.y(), p.z(), s};
    }
    // 不要在上一个循环就添加，应该等内存地址完全确定后再添加
    for (size_t i = 0; i < vecSim3Pose.size(); ++i) {
        problem.AddParameterBlock(vecSim3Pose[i].data(), 8,
                                  sim3PoseParameterization);
    }
    problem.SetParameterBlockConstant(vecSim3Pose[0].data());

    for (size_t i = 0; i < sT12Constraint.size() - 1; ++i) {
        // 添加帧间相对约束
        ceres::CostFunction* cost =
            new RelativeConstraintResidual(sT12Constraint[i]);
        problem.AddResidualBlock(cost, nullptr, vecSim3Pose[i].data(),
                                 vecSim3Pose[i + 1].data());
    }
    // 最后一帧是闭环约束
    ceres::CostFunction* cost =
        new RelativeConstraintResidual(sT12Constraint.back());
    problem.AddResidualBlock(cost, nullptr, vecSim3Pose[0].data(),
                             vecSim3Pose.back().data());

    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 500;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.preconditioner_type = ceres::SCHUR_JACOBI;
    options.minimizer_type = ceres::TRUST_REGION;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.num_threads = 1;
    // options.max_solver_time_in_seconds = 0.5;

    // 运行优化
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;

    for (size_t i = 0; i < vecSim3Pose.size(); ++i) {
        const auto& d = vecSim3Pose[i];
        const Eigen::Quaterniond q(d[0], d[1], d[2], d[3]);
        const Eigen::Vector3d p(d[4], d[5], d[6]);
        const double s = d[7];
        const double time = loopClosurePoseTwc[i].debugTimestamp_;
        loopClosurePoseTwc[i] = Sim3Pose(q, p, s);
        loopClosurePoseTwc[i].debugTimestamp_ = time;
    }

    of.open(kClosurePoseFilePath);
    for (size_t i = 0; i < loopClosurePoseTwc.size(); ++i) {
        // cout << "sTwc[" << i << "]: " << loopClosurePoseTwc[i].QwbString()
        //      << ", " << loopClosurePoseTwc[i].PwbString() << endl;
        of << loopClosurePoseTwc[i].DebugOutputPoseMessage() << endl;
    }
    of.close();

    return true;
}

void Optimizer::CalculateLastKFmeanDepth() {
    // 在最新关键帧被添加到滑窗内的时候调用
    if (window_.size() < 2) {
        lastKFmeanDepth_ = 0.0;
        return;
    }
    const KeyFrame* last = window_.back();
    double sumDepth = 0.;
    int num = 0;

    lock_guard<mutex> lock(globalOptFlwMutex);
    globalOptFlw.RemoveUselessLandmark();

    // 只有历史跟踪点才可能三角化成功
    for (size_t i = 0; i < globalOptFlw.historyLandmarkNum_; ++i) {
        auto lk = globalOptFlw.trackLandmark_[i];
        if (!lk->CanBeUseForOptimization() ||
            !lk->target_.count(const_cast<KeyFrame*>(last))) {
            continue;
        }

        const Eigen::Vector3d pc1 = lk->GetPc();
        if (pc1.z() < kMinSceneDepthInCamera) {
            lk->SetCanDelete();
            continue;
        }
        const Eigen::Vector3d pw = lk->host_->Twc_ * pc1;
        const Eigen::Vector3d pc2 = last->Tcw_ * pw;
        if (pc2.z() < kMinSceneDepthInCamera) {
            lk->SetCanDelete();
            continue;
        }
        sumDepth += pc2.z();
        ++num;
    }
    lastKFmeanDepth_ = sumDepth / num;
    cout << fmt::format("last kf id: {}, mean depth: {}\n", last->id_,
                        lastKFmeanDepth_);
}

void Optimizer::WriteDebugTrackLostStatus(const KeyFrame& curF) {
    unordered_map<const KeyFrame*, size_t> kf2Idx;
    for (size_t i = 0; i < window_.size(); ++i) {
        kf2Idx.insert({window_[i], i});
    }
    vector<vector<shared_ptr<Landmark>>> usefulMapPointEachKf(window_.size());
    vector<vector<cv::Point2f>> usefulObservationCurF(window_.size());
    for (size_t i = 0; i < usefulMapPointEachKf.size(); ++i) {
        usefulMapPointEachKf[i].reserve(500);
        usefulObservationCurF[i].reserve(500);
    }

    auto AssignLandmark = [&kf2Idx, &usefulMapPointEachKf,
                           &usefulObservationCurF](
                              const vector<shared_ptr<Landmark>>& lks,
                              const vector<cv::Point2f>& curFobv) {
        for (size_t i = 0; i < lks.size(); ++i) {
            if (lks[i]->NoUsed() || !lks[i]->CanBeUseForOptimization()) {
                continue;
            }
            const int idx = kf2Idx[lks[i]->host_];
            usefulMapPointEachKf[idx].emplace_back(lks[i]);
            usefulObservationCurF[idx].emplace_back(curFobv[i]);
        }
    };

    AssignLandmark(globalOptFlw.trackLandmark_, globalOptFlw.prevPts_);

    const int wDiff = window_[0]->debugGrayImg_.cols;
    const cv::Point2f pointDiff(wDiff, 0);
    const cv::Mat& debugImg2 = globalOptFlw.prevImg_;
    vector<string> colorKey;
    for (const auto& match : kColor) {
        colorKey.emplace_back(match.first);
    }
    constexpr int circleRadius = 3;
    cv::Mat showImg(window_[0]->debugGrayImg_.rows,
                    window_[0]->debugGrayImg_.cols * 2, CV_8UC3,
                    cv::Scalar{0, 0, 0});
    cv::Mat im1, im2;
    string videoPath = config->debugMessageSaveFolder;
    videoPath += "/track_lost_frame_message.avi";
    // 或者使用未压缩的格式（如果磁盘IO不是瓶颈）
    int fourcc = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
    int fps = 30;
    for (size_t i = 0; i < usefulMapPointEachKf.size(); ++i) {
        // 创建对比图像
        const cv::Mat& debugGrayImg = window_[i]->debugGrayImg_;
        cvtColor(debugGrayImg, im1, cv::COLOR_GRAY2BGR);
        cvtColor(debugImg2, im2, cv::COLOR_GRAY2BGR);
        im1.copyTo(showImg.colRange(0, debugGrayImg.cols));
        im2.copyTo(showImg.colRange(debugGrayImg.cols, showImg.cols));

        // 写入关键信息
        int start_text_row = 20;
        int step_text_row = 20;
        cv::putText(showImg,
                    fmt::format("window[{}], useful lk num: {}", i,
                                usefulMapPointEachKf[i].size()),
                    cv::Point(10, (start_text_row)), cv::FONT_ITALIC, 0.8,
                    kColor.at("red"), 1);
        cv::putText(showImg, fmt::format("curf id: {}", curF.id_),
                    cv::Point(10, (start_text_row += step_text_row)),
                    cv::FONT_ITALIC, 0.8, kColor.at("red"), 1);

        // 绘制匹配点
        for (size_t j = 0; j < usefulMapPointEachKf[i].size(); ++j) {
            std::shared_ptr<Landmark> const lk = usefulMapPointEachKf[i][j];
            const cv::Point2f& p1 = lk->GetHostFrameObvCV();
            const cv::Point2f& p2 = usefulObservationCurF[i][j] + pointDiff;
            const Vec3b& color = kColor.at(colorKey[rand() % kColor.size()]);
            // 画极线以查看匹配是否准确
            cv::circle(showImg, p1, circleRadius, color, 1);
            cv::circle(showImg, p2, circleRadius, color, 1);
            cv::line(showImg, p1, p2, color);
        }

        if (!debugTrackLostStatusVideoWriter_.isOpened()) {
            debugTrackLostStatusVideoWriter_.open(videoPath, fourcc, fps,
                                                  showImg.size(), true);
            if (debugTrackLostStatusVideoWriter_.isOpened()) {
                cout << "open track lost debug video at: " << videoPath << "\n";
            } else {
                cout << fmt::format(
                    "Error to open track lost debug video path: {}\n",
                    videoPath);
                return;
            }
        }
        debugTrackLostStatusVideoWriter_.write(showImg);
        cv::imshow(fmt::format("window[{}], useful lk num: {}", i,
                               usefulMapPointEachKf[i].size()),
                   showImg);
    }
    cv::waitKey();
    cv::destroyAllWindows();
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
    const Eigen::Vector2i& kp1 = lk1.GetHostFrameObvInt();
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

    const auto& _p = lk1.GetHostFrameObvInt();
    const string debugImgName = fmt::format("{}_{}", _p.x(), _p.y());
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

    const Eigen::Vector2d& kp1 = lk1.GetHostFrameObv();
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

    const auto& _p = lk1.GetHostFrameObvInt();
    const string debugImgName = fmt::format("{}_{}", _p.x(), _p.y());
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

    unordered_set<std::shared_ptr<Landmark>> aPoints, lPoints;
    {
        lock_guard<mutex> lock(KeyFrame::mutexForSyncView3Dstatus);
        KeyFrame::kfOn3Dshow.clear();
        // for (Landmark* p : optLandmark_) {
        //     if (p != nullptr && !aPoints.count(p) && !p->IsOutOfRange() &&
        //         p->ManySupport() && p->Converge()) {
        //         aPoints.insert(p);
        //     }
        // }

        for (int i = 0; i < static_cast<int>(window_.size()); ++i) {
            // for(int i = window_.size()-1; i < window_.size(); ++i) {
            KeyFrame* kf = window_[i];
            for (auto p : kf->landmark_) {
                if (p != nullptr && !aPoints.count(p) && !lPoints.count(p) &&
                    !p->CanBeDelete() && p->initialized_) {
                    lPoints.insert(p);
                }
            }

            KeyFrame::kfOn3Dshow.insert(kf);
        }
    }

    if (!aPoints.empty() || !lPoints.empty()) {
        {
            lock_guard<mutex> lockPoints(interaction->mutPoints);
            interaction->activePoints = aPoints;
            interaction->localPoints = lPoints;
        }
        ::ShowLocalMap(vTwc);
        cout << fmt::format("show {} KFs, aPoints size: {}, lPoints size: {}\n",
                            vTwc.size(), aPoints.size(), lPoints.size());
    } else {
        cerr << "wait for local map..." << endl;
        usleep(1000 * 1e3);
    }
}
