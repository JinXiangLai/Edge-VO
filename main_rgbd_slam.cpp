#include <unistd.h>
#include <fstream>
#include <memory>
#include <random>  // 随机数引擎
#include <stack>
#include <thread>

#include <opencv2/viz/vizcore.hpp>

#include "Config.h"
#include "Landmark.h"
#include "Optimizer.h"
#include "Pose.h"
#include "Utils.h"
#include "WheelCameraCalib.h"

using namespace std;
using namespace cv;

void Run(Optimizer* optimizer);

const cv::Point kViz3DWindowPos(1920 + 1920 / 2,
                                1080 - 100);  // 窗口左上角点在屏幕上的位置

void ResetStatus(Optimizer* optimizer, bool* isInitialized,
                 int* trackLostCount);

bool InitializeSecondKeyFramePose(const int findMatchNum,
                                  const double findMatchRatio,
                                  const shared_ptr<Camera>& cam, KeyFrame& curF,
                                  Optimizer& optimizer);

bool ConstructAndDecomposeEssentialMatrix(vector<Eigen::Vector4d> uv2obv,
                                          const shared_ptr<Camera>& cam,
                                          Pose& result);

bool CheckInitPose(const int selectNum, const shared_ptr<Camera>& cam,
                   const Pose& T12, vector<Eigen::Vector4d> uv2obv);

int main(int argc, char** argv) {

    // 读取程序参数
    //string configFilePath = "../config.yaml";
    string configFilePath = "../src/tum_config.yaml";

    if (argc < 2) {
        cerr << "[WARNING] Usage: ./main configFile[DEFAULT: " << configFilePath
             << "]" << endl;
    } else {
        configFilePath = string(argv[1]);
        cerr << "[INFO] configFile: " << configFilePath << endl;
    }

    Config _config(configFilePath);
    config = &_config;
    InteractionParam _visualizeParam;
    interaction = &_visualizeParam;
    viz::Viz3d window("Local Map Viewer");
    window.setWindowPosition(kViz3DWindowPos);
    interaction->window = &window;
    InitColor();

    const int firstImgIdx = config->firstImgIdx;
    // 读取外部数据
    vector<string> vstrImages;
    vector<double> vTimeStamps;
    vector<string> vDepthImgs;
    vector<double> vDepthImgTimes;

    // TODO:需要将轮速系转换为相机系，所以倒不如直接在ORBSLAM3下的框架进行开发呢！！！
    vector<Eigen::Matrix<double, 8, 1>> vPriorPose;
    if (config->model == "pinhole") {
        LoadImages(config->dataDir, vstrImages, vTimeStamps, ".png");
        if (config->useDepthImage || config->initWithTrueDepth ||
            config->debugWithTrueDepthImage) {
            LoadImages(config->dataDir, vDepthImgs, vDepthImgTimes, ".png",
                       true);
        }
    } else {
        LoadImages(config->dataDir, vstrImages, vTimeStamps);
    }
    LoadPriorOdom(config->dataDir, vPriorPose);

    for (size_t i = 1; i < vTimeStamps.size(); ++i) {
        Assert(vTimeStamps[i] - vTimeStamps[i - 1] > 0,
               "Check img timestamp error!!!");
    }
    for (size_t i = 1; i < vPriorPose.size(); ++i) {
        //cout << fixed << vPriorPose[i][0] << " | " << vPriorPose[i-1][0] << endl;
        Assert(vPriorPose[i][0] >= vPriorPose[i - 1][0],
               "Check odom timestamp error!!!");
    }

    WheelCameraCalib calib(config->Qcg, config->Pcg, config->wheelRadius);
    shared_ptr<Camera> cam = make_shared<Camera>(config);
    Optimizer optimizer(cam);

    KeyFrame lastF, lastLastF;

    thread* viewerThread = nullptr;

    bool isInitialized = false;
    int trackLostCount = 0;
    vector<KeyFrame*>& win = optimizer.window_;
    stack<KeyFrame>
        unmappedFrame;  // 由于当前把每一帧都用来更新深度，所以不需要像lsd slam那样保留一些帧

    const chrono::steady_clock::time_point& tStart =
        chrono::steady_clock::now();
    for (size_t i = firstImgIdx; i < vTimeStamps.size(); ++i) {
        Mat img;
        Pose Twc;
        GetImageAndPose(i, vstrImages, vTimeStamps, vPriorPose, calib, img,
                        Twc);
        Mat depthImg;
        if (config->useDepthImage || config->debugWithTrueDepthImage ||
            (config->initWithTrueDepth && !isInitialized)) {
            bool success = GetDepthImage(vTimeStamps[i], vDepthImgs,
                                         vDepthImgTimes, depthImg);
            if (!success) {
                cout << "Get depth at: " << to_string(vTimeStamps[i])
                     << " Failed!" << endl;
                continue;
            }
        }
        cout << i << " th cur img timestamp: " << to_string(vTimeStamps[i])
             << endl;

        KeyFrame curF(img, Twc, cam, i, config->pyrLevel);
        if (config->useDepthImage || config->debugWithTrueDepthImage) {
            curF.depthImage_ = depthImg;
        }

        if (win.empty()) {
            KeyFrame* initFrame = new KeyFrame(curF);
            // 首帧设置为单位矩阵
            initFrame->SetTwc(Pose(), true);
            initFrame->depthImage_ = depthImg;
            lastF = *initFrame;
            optimizer.AddOneKeyFeame(initFrame);
            interaction->visualLastKF = *win.back();

            if (config->debugShowOnlineResult3D && !viewerThread) {
                viewerThread = new thread(Run, &optimizer);
            }
            cout << "Set initFrame with frame id: " << i << endl;
            continue;  // 认为初始化完毕
        }

        // 使用KF更新当前帧的pose
        const Pose Twc2 = curF.priorTwc_;
        Pose Twc1, Tc1c2;
        Twc1 = win.back()->priorTwc_;
        Tc1c2 = Twc1.Inverse() * Twc2;
        curF.SetTwc(win.back()->Twc_ * Tc1c2);

        const double trans = (win.back()->priorTwc_.Inverse() * curF.priorTwc_)
                                 .t_wb_.head(2)
                                 .norm();

        // 显示线程使用
        //interaction->visualCurF = curF;
        interaction->visualCurFinit = curF;
        int findMatchNum = 0;
        if (!isInitialized) {
            // 初始化深度图
            lastLastF = lastF;
            lastF = curF;
            const double findMatchRatio =
                win.back()->TrackWithOpticalFlow(curF, findMatchNum);
            cout << "Initializing findMatchRatio, trans: " << findMatchRatio
                 << ", " << trans << endl;
            if ((trans > 0.0 &&
                 InitializeSecondKeyFramePose(findMatchNum, findMatchRatio, cam,
                                              curF, optimizer)) ||
                config->useDepthImage) {
                // 初始化深度图已经生成，后续需要对每一帧进行深度图传播
                isInitialized = true;
                cout << "\n******\nInitialized!\n******\n";
                // TODO: 初始化，首帧固定为单位阵，计算当前帧位姿
                optimizer.AddOneKeyFeame(new KeyFrame(curF));
            } else if (findMatchRatio < 0.8 || findMatchNum < 200) {
                cout << "Few match to initialize! Reset!" << endl;
                ResetStatus(&optimizer, &isInitialized, &trackLostCount);
            }

            continue;
        }

        // Step: 利用当前帧更新深度图

        // // TODO: 图像存在运动模糊时，会导致landmark, pose估计出异常值，
        // // 导致sliding window optimization优化崩溃：可仅优化pose而不优化landmark

        // step2: 利用当前帧更新landmark depth，depth与host frame绑定
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        const double findMatchRatio =
            win.back()->TrackWithOpticalFlow(curF, findMatchNum);
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();

        // step1: 优化当前帧pose
        Pose Ttemp = GetPredictPose(lastF, lastLastF);
        curF.SetTwc(Ttemp, true);
        bool trackLocalMapLow = false;
        const bool trackOk = optimizer.TrackLocalMap(
            &curF, trackLocalMapLow);  // TODO: 问题是这里的pose估计不准
        chrono::steady_clock::time_point t3 = chrono::steady_clock::now();

        bool debugReset =
            false && static_cast<int>(win.size()) == config->maxKFnumInWindow &&
            i % 150 == 0;  // debug
        if (!trackOk || debugReset) {
            ++trackLostCount;
            if (static_cast<int>(win.size()) < config->maxKFnumInWindow ||
                trackLostCount > 5 || debugReset)
                cout << fmt::format(
                            "track lost time: {}, sliding window size: {}, "
                            "debugReset: {}",
                            trackLostCount, win.size(), debugReset)
                     << endl;
            ResetStatus(&optimizer, &isInitialized, &trackLostCount);
            continue;
        }
        trackLocalMapLow = trackLocalMapLow && trackOk;
        //trackLocalMapLow = false; // 强制不使用

        // 显示线程更新使用
        interaction->visualCurF = curF;  // 记录优化pose后的当前帧
        // step3: 剔除地图外点, TODO: 应该使用融合而不是剔除策略！！！
        // optimizer.CullingErrorLandmark(&curF);

        // 当跟踪成功的点数量少于一定比例且运动满足阈值时，生成新的KF
        // Step: 当前帧选为新关键帧，
        // step1：追踪landmark，能够产生2D-2D的数据关联
        // step2：为剩余的edge point产生的landmark
        // 当前帧已经无法找到足够的匹配，需要创建新关键帧避免极线过长
        const bool caseOptflowTrackLow =
            (findMatchRatio < 0.7 || findMatchNum < 500);

        const Pose T12 = win.back()->Tcw_ * curF.Twc_;
        const double horDist = T12.t_wb_.head(2).norm();
        const double meanDepth = optimizer.GetLastKFmeanDepth();
        const bool caseMoveBaselineLOng =
            meanDepth > 0 && horDist > meanDepth * 0.5;

        // 必须保证当前KF收敛足够多的点了
        cout << fmt::format(
            "Need KF check: findMatchRatio:{:.1f}, findMatchNum: {}, "
            "trans "
            "dist: {:.2f}, "
            "rot ang: {:.1f}deg\n",
            findMatchRatio, findMatchNum, T12.t_wb_.head(2).norm(),
            Quat2RPY(T12.q_wb_).norm() * kRad2Deg);

        // 检验地图点跟踪效果，光流跟踪效果和运行基线
        if (((trackLocalMapLow || caseOptflowTrackLow) &&
             horDist > 0.05 * meanDepth) ||
            caseMoveBaselineLOng) {
            cout << fmt::format(
                "add kf case: trackLocalMapLow: {}, caseOptflowTrackLow: {}, "
                "caseMoveBaselineLOng: {}\n",
                trackLocalMapLow, caseOptflowTrackLow, caseMoveBaselineLOng);
            {
                static bool first = true;
                ofstream f;
                const string name = "generate_KF_case.csv";
                if (first) {
                    f.open(name.c_str(), ios::out);
                    first = false;
                    f << "#timestamp, qw, qx, qy, qz, x, y, z" << endl;
                    f.close();
                }
                f.open(name.c_str(), ios::app);
                //f << "(" <<case1 << " || " << case3 << " || " << case4 << " || " << case5 << ") && " << case2 << endl;
                const Eigen::Quaterniond& q = curF.priorTwc_.q_wb_;
                const Eigen::Vector3d& p = curF.priorTwc_.t_wb_;
                f << to_string(vTimeStamps[i]) << ", " << q.w() << ", " << q.x()
                  << ", " << q.y() << ", " << q.z() << ", " << p.x() << ", "
                  << p.y() << ", " << p.z() << endl;
                f.close();
            }

            // 可视化滑窗内点云
            // ShowPointCloud(curF.landmark_);
            // optimizer.ShowLocalMap(nullptr);

            optimizer.AddOneKeyFeame(new KeyFrame(curF));
            // TODO：当前帧被选为关键帧时，需要进行多帧的局部BA优化，因此需要添加互观测
            cout << "Add new keyframe id: " << curF.id_ << "\n";
            interaction->visualLastKF = *win.back();

        } else if (viewerThread != nullptr) {
            // delete curF; // 释放非KF内存
            usleep(10 * 1000);
        }

        lastLastF = lastF;
        lastF = curF;
        interaction->trajectory.push_back({curF.Twc_.t_wb_});

        cout << fmt::format(
            "TrackWithOpticalFlow spend:{:.3f}ms, TrackLocalMap spend: "
            "{:.3f}ms\n\n",
            ChronoMillisecTimeDuration(t1, t2),
            ChronoMillisecTimeDuration(t2, t3));

        while (interaction->stepBystep) {
            // 当前循环跑完，不需要再修改i
            usleep(100 * 1000);
        }
    }

    const chrono::steady_clock::time_point& tEnd = chrono::steady_clock::now();
    cout << fmt::format(
        "process {} images of sequence {}, total spend:{:.3f}s, sequence "
        "record duration: {:.3f}s\n",
        vTimeStamps.size() - firstImgIdx,
        config->dataDir.substr(config->dataDir.find_last_of('/') + 1),
        ChronoMillisecTimeDuration(tStart, tEnd) * 1e-3,
        vTimeStamps.back() - vTimeStamps[firstImgIdx]);

#if defined(WRITE_MATCH_PAIR_IMAGE)
    if (KeyFrame::debugVideoWriter.isOpened()) {
        KeyFrame::debugVideoWriter.release();
    }
    if (KeyFrame::debugTriangulateWriter.isOpened()) {
        KeyFrame::debugTriangulateWriter.release();
    }
#endif

    if (optimizer.debugTrackLostStatusVideoWriter_.isOpened()) {
        // 显式调用release才可写入视频
        optimizer.debugTrackLostStatusVideoWriter_.release();
    }

    if (config->debugShowOnlineResult3D) {
        viewerThread->join();
        delete viewerThread;
    }

    return 0;
}

void Run(Optimizer* optimizer) {
    while (1) {
        if (config->debugShowGlobalMap) {
            interaction->ShowGlobalMapPoint();
        } else if (!optimizer->window_.empty()) {
            optimizer->ShowLocalMap();
        }
        usleep(10 * 1000);
    }
}

void ResetStatus(Optimizer* optimizer, bool* isInitialized,
                 int* trackLostCount) {
    cout << fmt::format("Reset vo system!!!") << endl;
    *isInitialized = false;
    optimizer->window_.clear();
    KeyFrame::Tc0w = Pose();
    KeyFrame::kfOn3Dshow.clear();
    KeyFrame::optFlw.Reset();
    *trackLostCount = 0;
    interaction->trajectory.clear();
}

bool InitializeSecondKeyFramePose(const int findMatchNum,
                                  const double findMatchRatio,
                                  const shared_ptr<Camera>& cam, KeyFrame& curF,
                                  Optimizer& optimizer) {
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
    if (ConstructAndDecomposeEssentialMatrix(uv2obv, cam, result)) {
        curF.SetTwc(result);
        cout << fmt::format("Initialize succeed!, hor trans: {:.2f}m\n", trans);
        return true;
    }

    cout << fmt::format("Initialize failed!, hor trans: {:.2f}m\n", trans);
    return false;
}

bool ConstructAndDecomposeEssentialMatrix(vector<Eigen::Vector4d> uv2obv,
                                          const shared_ptr<Camera>& cam,
                                          Pose& result) {
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
        std::random_device rd;
        std::default_random_engine rng(rd());
        std::shuffle(uv2obv.begin(), uv2obv.end(), rng);

        constexpr int kSelectNum = 15;  // 随机选取N个点求解本质矩阵E
        Eigen::Matrix<double, kSelectNum, 9> A;
        A.setZero();
        const size_t selectStep = uv2obv.size() / kSelectNum;
        int useNum = 0;
        for (size_t j = 0; useNum < kSelectNum; j += selectStep) {
            const Eigen::Vector2d& p1 = uv2obv[j].head(2);
            const Eigen::Vector2d p2 = uv2obv[j].tail(2);
            const Eigen::Vector3d pn1 = cam->InverseProject(p1);
            const Eigen::Vector3d pn2 = cam->InverseProject(p2);
            const double x1 = pn1.x(), y1 = pn1.y(), x2 = pn2.x(), y2 = pn2.y();
            A.row(useNum) << x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2,
                1.0;
            ++useNum;
        }
        cout << "matrixA[" << i << "]:\n" << A << "\n";

        // SVD分解A，最小奇异值对应的特征向量即为e向量
        // Eigen::JacobiSVD<Eigen::Matrix<double, kSelectNum, 9>> svd(
        //     A, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::JacobiSVD<Eigen::Matrix<double, kSelectNum, 9>> svd(
            A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        const Eigen::Matrix<double, 9, 1> singValue = svd.singularValues();
        cout << "A matrix singValue: " << singValue.transpose() << "\n";
        Eigen::Matrix<double, 9, 1> e = svd.matrixV().col(8);
        // e.normalize();  // 归一化
        cout << "e: " << e.transpose() << "\n";
        Eigen::Matrix<double, 3, 3> essentialMatrix;
        essentialMatrix << e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8];
        cout << "essentialMatrix:\n" << essentialMatrix << "\n";

        // SVD分解E矩阵，并分解出旋转和平移
        // Eigen::JacobiSVD<Eigen::Matrix3d> svd2(
        //     essentialMatrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::JacobiSVD<Eigen::Matrix3d> svd2(
            essentialMatrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix<double, 3, 1> s2 = svd2.singularValues();
        cout << "essentialMatrix singValue: " << s2.transpose() << "\n";
        Eigen::Matrix3d S = Eigen::Matrix3d::Zero();
        // const double sigma = svd2.singularValues().head(2).sum() * 0.5;
        // S.diagonal().head(2) << sigma, sigma;
        S.diagonal().head(2) << 1.0, 1.0;
        // S.diagonal() = s2;

        Eigen::Matrix3d Rpi_2, R_n_pi_2;
        Rpi_2 << 0, -1, 0, 1, 0, 0, 0, 0, 1;
        R_n_pi_2 << 0, 1, 0, -1, 0, 0, 0, 0, 1;
        const Eigen::Matrix3d U2 = svd2.matrixU();
        const Eigen::Matrix3d V2 = svd2.matrixV();
        // 一共有四种组合，区别在于t可以取负号
        Eigen::Matrix3d R1 = U2 * Rpi_2.transpose() * V2.transpose();
        Eigen::Matrix3d R2 = U2 * R_n_pi_2.transpose() * V2.transpose();
        const Eigen::Vector3d t1 =
            SkewSymmetric2Vector(U2 * Rpi_2 * S * U2.transpose());
        const Eigen::Vector3d t2 =
            SkewSymmetric2Vector(U2 * R_n_pi_2 * S * U2.transpose());
        cout << fmt::format("det(R1): {:.1f}, det(R2):{:.1f}\n",
                            R1.determinant(), R2.determinant());
        // 修正反射矩阵
        if (R1.determinant() < 0) {
            R1 = -R1;
        }
        if (R2.determinant() < 0) {
            R2 = -R2;
        }
        vector<Eigen::Matrix3d> Rs{R1, R1, R2, R2};
        vector<Eigen::Vector3d> ts{t1, -t1, t2, -t2};
        if (t1.head(2).norm() / t1.norm() < 0.9) {
            cout << "Warning, horizontail move may small, trans: "
                 << t1.transpose() << "\n";
        }

        vector<int> usefulDepth(Rs.size(), 0);

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
                if (GetHostFrameObservationInvDepth(p1, p2, cam->Kinv_[0], T12,
                                                    idepth1)) {
                    depth = 1.0 / idepth1;
                }
                depthInfo.append(fmt::format("{:.2f} ", depth));
                if (depth > kMinSceneDepthInCamera) {
                    ++usefulDepth[k];
                }
            }
            // cout << fmt::format("{}\n{}", depthInfo, poseInfo);
            ++useNum;
        }

        cout << fmt::format("useful depth count: {}, {}, {}, {}\n",
                            usefulDepth[0], usefulDepth[1], usefulDepth[2],
                            usefulDepth[3]);
        int maxId = 0;
        for (size_t l = 0; l < 4; ++l) {
            if (usefulDepth[l] > usefulDepth[maxId]) {
                maxId = l;
            }
        }
        cout << fmt::format("maxId: {}, max positive num: {}\n", maxId,
                            usefulDepth[maxId]);
        if (usefulDepth[maxId] > static_cast<int>(kSelectNum * 0.9)) {
            Pose initT12(Eigen::Quaterniond(Rs[maxId]), ts[maxId]);
            // initT12.t_wb_.normalize(); // 不能归一化？
            if (CheckInitPose(kSelectNum, cam, initT12, uv2obv)) {
                result = initT12;
                cout << "Initialize succeed, result: " << result << "\n";
                return true;
            }
        }
    }

    return false;
}

bool CheckInitPose(const int selectNum, const shared_ptr<Camera>& cam,
                   const Pose& T12, vector<Eigen::Vector4d> uv2obv) {
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
                                             uv2obv[i].tail(2), cam->Kinv_[0],
                                             T12, idepth1)) {
            sumProjectErrorSelectPoint += 1e9;
            ids[row] = i;
            ++row;
            projResidualInfo.append(fmt::format(" {:.1f}", 1e9));
            continue;
        }

        const Eigen::Vector3d p1 =
            cam->InverseProject(uv2obv[i].head(2)) / idepth1;
        const Eigen::Vector3d pc2 = T21 * p1;
        const Eigen::Vector2d p2 = cam->Project2PixelPlane(pc2);
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

    if (meanResidual < 12.0) {
        return true;
    }

    return false;
}
