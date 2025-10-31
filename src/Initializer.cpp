#include "Initializer.h"
#include <fmt/core.h>
#include <random>
#include "Utils.h"

using namespace std;

Initializer::Initializer(const shared_ptr<Camera>& cam, Optimizer* optimizer)
    : cam_(cam), optimizer_(optimizer) {}

Initializer::~Initializer() {
    if (debugInitTrackWriter_.isOpened()) {
        debugInitTrackWriter_.release();
    }
}

bool Initializer::InitializeSecondKeyFramePose(const int findMatchNum,
                                               const double findMatchRatio,
                                               KeyFrame& curF) {
    constexpr int kMinMatchFeatureNum = 100;
    constexpr double kMinParallax = 3;
    if (findMatchNum < kMinMatchFeatureNum) {
        return false;
    }

    const KeyFrame::OpticalFlowStruct optFlw = KeyFrame::optFlw;
    Eigen::Vector2d sumParallax(0, 0);  // 沿z轴时，理论上视差会抵消
    for (size_t i = 0; i < optFlw.trackLandmark_.size(); ++i) {
        const Eigen::Vector2d& p1 = optFlw.trackLandmark_[i]->uv_;
        const cv::Point2f& p2 = optFlw.prevPts_[i];
        const Eigen::Vector2d parallax(p1.x() - p2.x, p1.y() - p2.y);
        sumParallax += parallax;
    }
    const double meanParallax =
        sumParallax.norm() / optFlw.trackLandmark_.size();
    if (meanParallax < kMinParallax) {
        cout << fmt::format("parallax: {:.1f} too small\n", meanParallax);
        return false;
    } else {
        cout << fmt::format("parallax: {:.1f} can used for initializing\n",
                            meanParallax);
    }

    vector<Eigen::Vector4d> uv2obv;
    uv2obv.reserve(optFlw.trackLandmark_.size());
    for (size_t i = 0; i < optFlw.trackLandmark_.size(); ++i) {
        const Eigen::Vector2d& uv = optFlw.trackLandmark_[i]->uv_;
        const cv::Point2f obv = optFlw.prevPts_[i];
        uv2obv.emplace_back(uv.x(), uv.y(), obv.x, obv.y);
    }

    const double trans =
        (optFlw.trackLandmark_[0]->host_->priorTwc_.Inverse() * curF.priorTwc_)
            .t_wb_.head(2)
            .norm();
    Pose result;
    // if (ConstructAndDecomposeEssentialMatrix(uv2obv, result)) {
    // if (ConstructAndDecomposeEssentialMatrixOpenCV(uv2obv, result)) {
    if (ConstructAndDecomposeEssentialMatrixNormPoint(uv2obv, result)) {
        result.t_wb_ =
            result.t_wb_.normalized() * curF.Tcw_.t_wb_.norm();  // 仅做debug
        const Pose Tdiff = result * curF.Tcw_;
        cout << "result error: " << Tdiff << endl;
        curF.SetTwc(result);
        cout << fmt::format("Initialize succeed!, hor trans: {:.2f}m\n", trans);
        return true;
    }

    cout << fmt::format("Initialize failed!, hor trans: {:.2f}m\n", trans);
    return false;
}

bool Initializer::ConstructAndDecomposeEssentialMatrix(
    vector<Eigen::Vector4d>& uv2obv, Pose& result) {
    // 根据对极线约束，首帧为w系
    // s1 * pn1 = s2 * Rwc2 * pn2 + t_wc2
    // s1 * [t_wc2]x * pn1 = s2 * [t_wc2]x * Rwc2 * pn2
    // pn1.T * [t_wc2]x * Rwc2 * pn2 = 0
    //                [e11, e12, e13]   [x2]
    // [x1, y1, z1] * [e21, e22, e23] * [y2]
    //                [e31, e32, e33]   [z2]
    //
    // 分析矩阵可知，pn1中每一列会与E矩阵的对应行相乘
    // 同时，pn2每一行会与E矩阵的每一列对应相乘，由此写出矩阵乘法                                [x2]
    // [x1*e11 + y1*e21 + z1*e31, x1*e12 + y1*e22 + z1*e32, x1*e13 + y1*e23 + z1*e33] * [y2]
    //                                                                                  [z2]
    // 得出结果为：
    // (x1*x2*e11 + y1*x2*e21 + z1*x2*e31) + (x1*y2*e12 + y1*y2*e22 + z1*y2*e32) + (x1*z2*e13 + y1*z2*e23 + z1*z2*e33)
    // 将上式E矩阵按行优先排列，可以写出如下等式：
    // [x1*x2, x1*y2, x1*z2, y1*x2, y1*y2, y1*z2, z1*x2, z1*y2, z1*z2] * e.T = 0 [1x1]标量
    // 由于z1=z2=1，故简化为：
    // [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1] * e.T = 0

    const KeyFrame::OpticalFlowStruct optFlw = KeyFrame::optFlw;
    constexpr int kMaxIterateTime = 20;
    for (size_t i = 0; i < kMaxIterateTime; ++i) {
        random_device rd;
        default_random_engine rng(rd());
        shuffle(uv2obv.begin(), uv2obv.end(), rng);

        constexpr int kSelectNum = 8;  // 随机选取N个点求解本质矩阵E
        Eigen::Matrix<double, kSelectNum, 9> A;
        A.setZero();
        const size_t selectStep = (uv2obv.size() - 1) / kSelectNum;
        int useNum = 0;
        GenerateDebugImage(optFlw.prevImg_);
        vector<cv::Point2f> pts1, pts2;
        // vector<Eigen::Vector2d> ps1, ps2;
        for (size_t j = 0; useNum < kSelectNum; j += selectStep) {
            const Eigen::Vector2d& p1 = uv2obv[j].head(2);
            const Eigen::Vector2d p2 = uv2obv[j].tail(2);
            DrawMatchPoint(p1, p2);
            const Eigen::Vector3d pn1 = cam_->InverseProject(p1);
            const Eigen::Vector3d pn2 = cam_->InverseProject(p2);
            const double x1 = pn1.x(), y1 = pn1.y(), x2 = pn2.x(), y2 = pn2.y();
            A.row(useNum) << x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2,
                1.0;
            ++useNum;
            pts1.emplace_back(p1.x(), p1.y());
            pts2.emplace_back(p2.x(), p2.y());
        }
        // cout << "matrixA[" << i << "]:\n" << A << "\n";

        // SVD分解A，最小奇异值对应的特征向量即为e向量
        // Eigen::JacobiSVD<Eigen::Matrix<double, kSelectNum, 9>> svd(
        //     A, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::JacobiSVD<Eigen::Matrix<double, kSelectNum, 9>> svd(
            A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        cv::Mat cvE = cv::findEssentialMat(pts2, pts1, Eigen2CVmat(cam_->K_[0]),
                                           cv::LMEDS);
        cout << "A matrix row: " << A.rows()
             << ", singValue: " << svd.singularValues().transpose() << "\n";
        Eigen::Matrix<double, 9, 1> e = svd.matrixV().col(8);
        cout << "e: " << e.transpose() << "\n";
        Eigen::Matrix<double, 3, 3> essentialMatrix;
        essentialMatrix << e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8];
        cout << "essentialMatrix:\n" << essentialMatrix << "\n";
        cout << "essential matrix from cv:\n" << cvE << "\n";
        // SVD分解E矩阵，并分解出旋转和平移
        // Eigen::JacobiSVD<Eigen::Matrix3d> svd2(
        //     essentialMatrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::JacobiSVD<Eigen::Matrix3d> svd2(
            essentialMatrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix<double, 3, 1> s2 = svd2.singularValues();
        cout << "essentialMatrix singValue: " << s2.transpose() << "\n";

        Eigen::Matrix3d S = Eigen::Matrix3d::Zero();
        const double sigma = svd2.singularValues().head(2).sum() * 0.5;
        S.diagonal().head(2) << sigma, sigma;
        S.diagonal().head(2) << 1.0, 1.0;
        const Eigen::Matrix3d refineE =
            svd2.matrixU() * S * svd2.matrixV().transpose();
        Eigen::JacobiSVD<Eigen::Matrix3d> svd3(
            refineE, Eigen::ComputeFullU | Eigen::ComputeFullV);
        cout << "refineE:\n" << refineE << "\n";
        cout << "refineE singValue: " << svd3.singularValues().transpose()
             << "\n";

        Eigen::Matrix3d Rpi_2, R_n_pi_2;
        Rpi_2 << 0, -1, 0, 1, 0, 0, 0, 0, 1;
        R_n_pi_2 << 0, 1, 0, -1, 0, 0, 0, 0, 1;
        const Eigen::Matrix3d U2 = svd3.matrixU();
        const Eigen::Matrix3d V2 = svd3.matrixV();
        // 一共有四种组合，区别在于t可以取负号
        // Eigen::Matrix3d R1 = U2 * Rpi_2.transpose() * V2.transpose();
        // Eigen::Matrix3d R2 = U2 * R_n_pi_2.transpose() * V2.transpose();
        // const Eigen::Vector3d t1 =
        //     SkewSymmetric2Vector(U2 * Rpi_2 * S * U2.transpose());
        // const Eigen::Vector3d t2 =
        //     SkewSymmetric2Vector(U2 * R_n_pi_2 * S * U2.transpose());
        Eigen::Matrix3d R1 = U2 * Rpi_2 * V2.transpose();
        Eigen::Matrix3d R2 = U2 * R_n_pi_2 * V2.transpose();
        const Eigen::Vector3d t1 = U2.col(2);
        const Eigen::Vector3d t2 = -U2.col(2);
        cout << fmt::format("det(R1): {:.1f}, det(R2):{:.1f}\n",
                            R1.determinant(), R2.determinant());
        // 修正反射矩阵
        if (R1.determinant() < 0) {
            R1 = -R1;
        }
        if (R2.determinant() < 0) {
            R2 = -R2;
        }
        // 2. 分解本质矩阵得到R,t
        cv::Mat cvR, cv_t, mask;
        int inliers = recoverPose(cvE, pts2, pts1, Eigen2CVmat(cam_->K_[0]),
                                  cvR, cv_t, mask);
        // opencv是计算T21，因此输入特征点要倒序
        cout << "OpenCV recoverPose found " << inliers << " inliers" << endl;
        cout << "cvR: " << cvR << endl;
        cout << "cv_t: " << cv_t.t() << ", norm: " << cv::norm(cv_t) << endl;

        cout << "R1:\n"
             << R1 << "\nR2:\n"
             << R2 << "\nt1: " << t1.transpose() << ", norm: " << t1.norm()
             << ", t2: " << t2.transpose() << ", norm: " << t2.norm() << "\n";
        // const Eigen::Matrix3d Rres = CVmat2Eigen(cvR);
        // Eigen::Quaterniond q(Rres);
        // Pose initT12(q, CVmat2Eigen(cv_t));
        // result = initT12;
        // return true;

        vector<Eigen::Matrix3d> Rs{R1, R1, R2, R2};
        vector<Eigen::Vector3d> ts{t1, -t1, t2, -t2};
        if (t1.head(2).norm() / t1.norm() < 0.9) {
            cout << "Warning, horizontail move may small, trans: "
                 << t1.transpose() << "\n";
        }

        // vector<int> usefulDepth(Rs.size(), 0);
        vector<pair<int, int>> id2UsefulDepth{{0, 0}, {1, 0}, {2, 0}, {3, 0}};

        // 找出正确的旋转和平移
        useNum = 0;
        for (size_t j = 0; useNum < kSelectNum; j += selectStep) {
            const Eigen::Vector2d& p1 = uv2obv[j].head(2);
            const Eigen::Vector2d p2 = uv2obv[j].tail(2);

            string depthInfo("depth result: ");
            string poseInfo("4 pose info:\n");
            for (size_t k = 0; k < 4; ++k) {
                double idepth1 = -1.0;
                double depth = -1.0;
                Pose T12(Eigen::Quaterniond(Rs[k]), ts[k]);
                stringstream ss;
                ss << T12;
                poseInfo.append(fmt::format("{}\n", ss.str()));
                if (GetHostFrameObservationInvDepth(p1, p2, cam_->Kinv_[0], T12,
                                                    idepth1)) {
                    depth = 1.0 / idepth1;
                }
                depthInfo.append(fmt::format("{:.2f} ", depth));
                if (depth > kMinSceneDepthInCamera) {
                    // ++usefulDepth[k];
                    ++id2UsefulDepth[k].second;
                }
            }
            // cout << fmt::format("{}\n{}", depthInfo, poseInfo);
            ++useNum;
        }

        cout << fmt::format("useful depth count: {}, {}, {}, {}\n",
                            id2UsefulDepth[0].second, id2UsefulDepth[1].second,
                            id2UsefulDepth[2].second, id2UsefulDepth[3].second);
        // cout << fmt::format("useful depth count: {}, {}, {}, {}\n",
        //                     id2UsefulDepth[0], id2UsefulDepth[1],
        //                     id2UsefulDepth[2], id2UsefulDepth[3]);
        sort(id2UsefulDepth.begin(), id2UsefulDepth.end(),
             [](const pair<int, int>& p1, const pair<int, int>& p2) {
                 return p1.second > p2.second;
             });

        if (id2UsefulDepth[0].second > static_cast<int>(kSelectNum * 0.9) &&
            id2UsefulDepth[1].second < static_cast<int>(kSelectNum * 0.5)) {
            const int maxId = id2UsefulDepth[0].first;
            cout << fmt::format(
                "maxId: {}, maxNum: {}, second maxId: {}, second maxNum: {}\n",
                maxId, id2UsefulDepth[0].second, id2UsefulDepth[1].first,
                id2UsefulDepth[1].second);

            Eigen::Quaterniond q(Rs[maxId]);
            Pose initT12(q, ts[maxId]);
            // initT12.t_wb_.normalize(); // 不能归一化？
            if (CheckInitPose(kSelectNum, initT12, uv2obv)) {
                result = initT12;
                cout << "Initialize succeed, result: " << result << "\n";
                return true;
            }
        }
    }

    return false;
}

bool Initializer::ConstructAndDecomposeEssentialMatrixOpenCV(
    vector<Eigen::Vector4d>& uv2obv, Pose& result) {
    random_device rd;
    default_random_engine rng(rd());
    shuffle(uv2obv.begin(), uv2obv.end(), rng);

    const int kSelectNum = min(
        200, static_cast<int>(uv2obv.size()));  // 随机选取N个点求解本质矩阵E
    const size_t selectStep = (uv2obv.size() - 1) / kSelectNum;
    vector<cv::Point2f> ps1, ps2;
    int useNum = 0;
    GenerateDebugImage(KeyFrame::optFlw.prevImg_);
    for (size_t j = 0; useNum < kSelectNum; j += selectStep) {
        const Eigen::Vector2d& p1 = uv2obv[j].head(2);
        const Eigen::Vector2d p2 = uv2obv[j].tail(2);
        DrawMatchPoint(p1, p2);
        ++useNum;
        ps1.emplace_back(p1.x(), p1.y());
        ps2.emplace_back(p2.x(), p2.y());
    }

    cv::Mat cvE =
        cv::findEssentialMat(ps2, ps1, Eigen2CVmat(cam_->K_[0]), cv::RANSAC);

    // 2. 分解本质矩阵得到R,t
    cv::Mat cvR, cv_t, mask;
    int inliers =
        recoverPose(cvE, ps2, ps1, Eigen2CVmat(cam_->K_[0]), cvR, cv_t, mask);
    cout << "OpenCV recoverPose found " << inliers << " inliers" << endl;
    cout << "cvR: " << cvR << endl;
    cout << "cv_t: " << cv_t.t() << ", norm: " << cv::norm(cv_t) << endl;

    const Eigen::Matrix3d Rres = CVmat2Eigen(cvR);
    Eigen::Quaterniond q(Rres);
    Pose initT12(q, CVmat2Eigen(cv_t));
    result = initT12;

    return inliers > static_cast<int>(ps1.size() * 0.75);
}

bool Initializer::CheckInitPose(const int selectNum, const Pose& T12,
                                vector<Eigen::Vector4d>& uv2obv) {
    const Pose T21 = T12.Inverse();
    double sumProjectErrorSelectPoint = 0.;
    const size_t selectStep = uv2obv.size() / selectNum;
    Eigen::VectorXi ids(selectNum);
    ids.setConstant(-1);
    int row = 0;
    string projResidualInfo("Check R,t project residual:");
    int useNum = 0;
    for (size_t i = 0; useNum < selectNum; i += selectStep) {
        double idepth1 = 0;
        ++useNum;
        if (!GetHostFrameObservationInvDepth(uv2obv[i].head(2),
                                             uv2obv[i].tail(2), cam_->Kinv_[0],
                                             T12, idepth1)) {
            sumProjectErrorSelectPoint += 1e9;
            ids[row] = i;
            ++row;
            projResidualInfo.append(fmt::format(" {:.1f}", 1e9));
            continue;
        }

        const Eigen::Vector3d p1 =
            cam_->InverseProject(uv2obv[i].head(2)) / idepth1;
        const Eigen::Vector3d pc2 = T21 * p1;
        if (pc2.z() < kMinSceneDepthInCamera) {
            sumProjectErrorSelectPoint += 1e9;
            projResidualInfo.append(fmt::format(" {:.1f}", 1e9));
            continue;
        }
        const Eigen::Vector2d p2 = cam_->Project2PixelPlane(pc2);
        const double residual = (p2 - uv2obv[i].tail(2)).norm();
        sumProjectErrorSelectPoint += residual;
        projResidualInfo.append(fmt::format(" {:.1f}", residual));
    }
    // TODO: 删除错误的id

    const double meanResidual = sumProjectErrorSelectPoint / selectNum;
    projResidualInfo.append(
        fmt::format("; sum residual: {:.1f}; mean residual: {:.1f}",
                    sumProjectErrorSelectPoint, meanResidual));
    cout << projResidualInfo << "\n";

    if (meanResidual < 5.0) {
        return true;
    }

    return false;
}

// 计算 Hartley 归一化变换 T，使点集中心在原点、平均距离为 sqrt(2)
Eigen::Matrix3d Initializer::ComputeNormalizationTransform(
    const vector<Eigen::Vector2d>& pts) {

    // return Eigen::Matrix3d::Identity(); // 验证不使用归一化时的效果

    const int N = static_cast<int>(pts.size());
    double cx = 0.0, cy = 0.0;
    for (const auto& p : pts) {
        cx += p.x();
        cy += p.y();
    }
    cx /= N;
    cy /= N;

    double meanDist = 0.0;
    for (const auto& p : pts) {
        double dx = p.x() - cx;
        double dy = p.y() - cy;
        meanDist += sqrt(dx * dx + dy * dy);
    }
    meanDist = (N > 0) ? meanDist / N : 1.0;

    double s = (meanDist > 1e-12) ? sqrt(2.0) / meanDist : 1.0;

    Eigen::Matrix3d T = Eigen::Matrix3d::Identity();
    T(0, 0) = s;
    T(1, 1) = s;
    T(0, 2) = -s * cx;
    T(1, 2) = -s * cy;
    return T;
}

// 应用 3x3 变换到 2D 点
vector<Eigen::Vector2d> Initializer::ApplyTransform(
    const vector<Eigen::Vector2d>& pts, const Eigen::Matrix3d& T) {
    vector<Eigen::Vector2d> out;
    out.reserve(pts.size());
    for (const auto& p : pts) {
        Eigen::Vector3d ph(p.x(), p.y(), 1.0);
        Eigen::Vector3d pn = T * ph;
        out.push_back({pn(0) / pn(2), pn(1) / pn(2)});
    }
    return out;
}

// 构造 A（N x 9），满足 x2^T E x1 = 0
Eigen::MatrixXd Initializer::BuildDesignMatrix(
    const vector<Eigen::Vector2d>& pts1, const vector<Eigen::Vector2d>& pts2) {
    const int N = static_cast<int>(pts1.size());
    Eigen::MatrixXd A(N, 9);
    for (int i = 0; i < N; ++i) {
        double u1 = pts1[i].x(), v1 = pts1[i].y();
        double u2 = pts2[i].x(), v2 = pts2[i].y();
        A(i, 0) = u2 * u1;
        A(i, 1) = u2 * v1;
        A(i, 2) = u2;
        A(i, 3) = v2 * u1;
        A(i, 4) = v2 * v1;
        A(i, 5) = v2;
        A(i, 6) = u1;
        A(i, 7) = v1;
        A(i, 8) = 1.0;
    }
    return A;
}

// 将 9 维向量重组为 3x3 矩阵（按行）
Eigen::Matrix3d Initializer::Vec9ToMat3(const Eigen::VectorXd& v) {
    Eigen::Matrix3d E;
    E << v(0), v(1), v(2), v(3), v(4), v(5), v(6), v(7), v(8);
    return E;
}

// 对 E 施加本质矩阵约束：两个相等非零奇异值，第三个为 0
Eigen::Matrix3d Initializer::EnforceEssentialConstraint(
    const Eigen::Matrix3d& E_initial) {
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(
        E_initial, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    Eigen::Vector3d S = svd.singularValues();

    // 保持尺度合理：取前两奇异值的平均
    double s = 0.5 * (S(0) + S(1));
    Eigen::Matrix3d S_new = Eigen::Matrix3d::Zero();
    S_new(0, 0) = s;
    S_new(1, 1) = s;

    // 保证 U、V 对应的旋转是正定的（可选）
    if (U.determinant() < 0)
        U.col(2) *= -1;
    if (V.determinant() < 0)
        V.col(2) *= -1;

    Eigen::Matrix3d E = U * S_new * V.transpose();
    return E;
}

// 计算残差（x2^T E x1）的 RMS
double Initializer::ComputeEpipolarRMS(const Eigen::Matrix3d& E,
                                       const vector<Eigen::Vector2d>& pts1,
                                       const vector<Eigen::Vector2d>& pts2) {
    double sum2 = 0.0;
    for (size_t i = 0; i < pts1.size(); ++i) {
        Eigen::Vector3d x1(pts1[i].x(), pts1[i].y(), 1.0);
        Eigen::Vector3d x2(pts2[i].x(), pts2[i].y(), 1.0);
        double r = x2.transpose() * E * x1;
        sum2 += r * r;
    }
    return sqrt(sum2 / pts1.size());
}

bool Initializer::ConstructAndDecomposeEssentialMatrixNormPoint(
    vector<Eigen::Vector4d>& uv2obv, Pose& result) {

    random_device rd;
    default_random_engine rng(rd());
    shuffle(uv2obv.begin(), uv2obv.end(), rng);

    const int kSelectNum = min(
        200, static_cast<int>(uv2obv.size()));  // 随机选取N个点求解本质矩阵E
    const size_t selectStep = (uv2obv.size() - 1) / kSelectNum;
    vector<Eigen::Vector2d> ps1, ps2;
    vector<cv::Point2f> pts1, pts2;
    pts1.reserve(kSelectNum);
    pts2.reserve(kSelectNum);
    ps1.reserve(kSelectNum);
    ps2.reserve(kSelectNum);
    int useNum = 0;
    GenerateDebugImage(KeyFrame::optFlw.prevImg_);
    for (size_t j = 0; useNum < kSelectNum; j += selectStep) {
        const Eigen::Vector2d& p1 = uv2obv[j].head(2);
        const Eigen::Vector2d p2 = uv2obv[j].tail(2);
        DrawMatchPoint(p1, p2);
        ++useNum;
        pts1.emplace_back(p1.x(), p1.y());
        pts2.emplace_back(p2.x(), p2.y());
        const Eigen::Vector3d pn1 = cam_->InverseProject(p1);
        const Eigen::Vector3d pn2 = cam_->InverseProject(p2);
        ps1.emplace_back(pn1.x(), pn1.y());
        ps2.emplace_back(pn2.x(), pn2.y());
    }

    // Hartley 归一化（在相机归一化平面上再做一次相似归一化，以提高数值稳定性）
    Eigen::Matrix3d T1 = ComputeNormalizationTransform(ps1);
    Eigen::Matrix3d T2 = ComputeNormalizationTransform(ps2);
    vector<Eigen::Vector2d> npts1 = ApplyTransform(ps1, T1);
    vector<Eigen::Vector2d> npts2 = ApplyTransform(ps2, T2);

    // 构造设计矩阵并估计初始 E'（在归一化坐标下）
    Eigen::MatrixXd A = BuildDesignMatrix(npts1, npts2);  // 计算T12
    Eigen::JacobiSVD<Eigen::MatrixXd> svdA(A, Eigen::ComputeFullV);
    Eigen::VectorXd evec = svdA.matrixV().col(8);  // 最小奇异值对应的右奇异向量
    Eigen::Matrix3d E_prime = Vec9ToMat3(evec);

    // 反归一化得到 E（满足 x2^T E x1 = 0）
    Eigen::Matrix3d E_initial = T2.transpose() * E_prime * T1;

    // 约束为本质矩阵
    Eigen::Matrix3d E_final = EnforceEssentialConstraint(E_initial);

    // 残差
    double rms_initial = ComputeEpipolarRMS(E_initial, ps1, ps2);
    double rms_final = ComputeEpipolarRMS(E_final, ps1, ps2);

    // 输出
    cout << fixed << setprecision(10);
    cout << "Essential matrix (init):\n" << E_initial << "\n";
    cout << "Essential matrix (constrained):\n";
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            cout << E_final(r, c) << (c == 2 ? '\n' : ' ');
        }
    }
    cout << "RMS residual before constraint: " << rms_initial << "\n";
    cout << "RMS residual after  constraint: " << rms_final << "\n";

    // 统一计算T21
    cv::Mat cvE =
        cv::findEssentialMat(pts1, pts2, Eigen2CVmat(cam_->K_[0]), cv::LMEDS);
    cout << "essential matrix cv:\n" << cvE << "\n";

    // 2. 分解本质矩阵得到R,t
    cv::Mat cvR, cv_t, mask;
    int inliers = recoverPose(Eigen2CVmat(E_final), pts1, pts2,
                              Eigen2CVmat(cam_->K_[0]), cvR, cv_t, mask);
    cout << "Norm point recoverPose found " << inliers << " inliers" << endl;
    cout << "normR: " << cvR << endl;
    cout << "norm_t: " << cv_t.t() << ", norm: " << cv::norm(cv_t) << endl;

    // inliers =
    //     recoverPose(cvE, pts1, pts2, Eigen2CVmat(cam_->K_[0]), cvR, cv_t, mask);
    // cout << "cv recoverPose found " << inliers << " inliers" << endl;
    // cout << "cvR: " << cvR << endl;
    // cout << "cv_t: " << cv_t.t() << ", norm: " << cv::norm(cv_t) << endl;
    cvR = cvR.clone().t();
    cv_t = -cvR * cv_t.clone();

    const Eigen::Matrix3d Rres = CVmat2Eigen(cvR);
    Eigen::Quaterniond q(Rres);
    Pose initT12(q, CVmat2Eigen(cv_t));
    result = initT12;

    return inliers > static_cast<int>(ps1.size() * 0.75);
}

void Initializer::GenerateDebugImage(const cv::Mat& curImg) {
    if (debugMatchImg_.empty()) {
        debugMatchImg_ =
            cv::Mat(curImg.rows, curImg.cols * 2, CV_8UC3, cv::Scalar{0, 0, 0});
    }
    cv::Mat im1, im2;
    cvtColor(optimizer_->window_.back()->debugGrayImg_, im1,
             cv::COLOR_GRAY2BGR);
    im1.copyTo(debugMatchImg_.colRange(0, curImg.cols));
    cvtColor(curImg, im2, cv::COLOR_GRAY2BGR);
    im2.copyTo(debugMatchImg_.colRange(curImg.cols, debugMatchImg_.cols));
}

void Initializer::DrawMatchPoint(const Eigen::Vector2d& kp1,
                                 const Eigen::Vector2d& kp2) {
    const int colorId = abs(rand()) % kColor.size();
    int idx = -1;
    cv::Vec3b color(0, 0, 0);
    for (const auto& c : kColor) {
        if (++idx == colorId) {
            color = c.second;
            break;
        }
    }
    int radius = 1;

    cv::Point p1(kp1.x(), kp1.y());
    cv::circle(debugMatchImg_, p1, radius, color, 1);
    cv::Point p2 = cv::Point(debugMatchImg_.cols * 0.5 + kp2.x(), kp2.y());
    cv::circle(debugMatchImg_, p2, radius, color, 1);
    cv::line(debugMatchImg_, p1, p2, color, 1);
}

void Initializer::PutText2DebugMatchImg(const string& info,
                                        const int writeRow) {
    cv::putText(debugMatchImg_, info, cv::Point(10, writeRow), cv::FONT_ITALIC,
                0.8, kColor.at("red"), 1);
}

void Initializer::WriteDebugInitImage() {

    string videoPath = config->debugMessageSaveFolder;
    videoPath += "/init_match_result.avi";
    // 或者使用未压缩的格式（如果磁盘IO不是瓶颈）
    int fourcc = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
    int fps = 30;
    debugInitTrackWriter_.open(videoPath, fourcc, fps, debugMatchImg_.size(),
                               true);
    if (!debugInitTrackWriter_.isOpened()) {
        cerr << "Open debug video path: " << videoPath << " failed" << endl;
    }
    cout << "Open debug video path: " << videoPath << endl;
    debugInitTrackWriter_.write(debugMatchImg_);
}