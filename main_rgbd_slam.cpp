#include <unistd.h>
#include <fstream>
#include <memory>
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

        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        chrono::steady_clock::time_point t3 = chrono::steady_clock::now();
        chrono::steady_clock::time_point t4 = chrono::steady_clock::now();

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
        cout << "win.size: " << win.size() << endl;

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
            if ((trans > 0.2 && (curF.id_ - win.back()->id_ > 30)) ||
                config->useDepthImage) {
                // 初始化深度图已经生成，后续需要对每一帧进行深度图传播
                isInitialized = true;
                cout << "\n******\nInitialized!\n******\n";
                // TODO: 初始化，首帧固定为单位阵，计算当前帧位姿
                optimizer.AddOneKeyFeame(new KeyFrame(curF));
            } else if (findMatchNum < 0.3) {
                cout << "Few match to initialize! Reset!" << endl;
                ResetStatus(&optimizer, &isInitialized, &trackLostCount);
            }

            continue;
        }

        // Step: 利用当前帧更新深度图

        // // TODO: 图像存在运动模糊时，会导致landmark, pose估计出异常值，
        // // 导致sliding window optimization优化崩溃：可仅优化pose而不优化landmark

        // step2: 利用当前帧更新landmark depth，depth与host frame绑定
        chrono::steady_clock::time_point t5 = chrono::steady_clock::now();
        const double findMatchRatio =
            win.back()->TrackWithOpticalFlow(curF, findMatchNum);
        chrono::steady_clock::time_point t6 = chrono::steady_clock::now();

        // step1: 优化当前帧pose
        Pose Ttemp = GetPredictPose(lastF, lastLastF);
        curF.SetTwc(Ttemp, true);
        bool trackLocalMapLow = false;
        const bool trackOk = optimizer.TrackLocalMap(
            &curF, trackLocalMapLow);  // TODO: 问题是这里的pose估计不准
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

        // TODO 1：利用跟踪结果更新当前帧pose，做当前帧和关键帧之间的BA优化
        // 这里暂时利用真值实现

        // TODO 2：利用跟踪帧进行必要的三角化(这个应该是在添加关键帧之后进行局部BA之后进行)
        chrono::steady_clock::time_point t7 = chrono::steady_clock::now();

        chrono::steady_clock::time_point t8 = chrono::steady_clock::now();

        chrono::steady_clock::time_point t9 = chrono::steady_clock::now();
        //cout << "Fuse depth spend " << chrono::duration<double>(t2 - t1).count() << "s" << endl;

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
        chrono::steady_clock::time_point t10, t11;

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

            t10 = chrono::steady_clock::now();
            optimizer.AddOneKeyFeame(new KeyFrame(curF));
            // TODO：当前帧被选为关键帧时，需要进行多帧的局部BA优化，因此需要添加互观测
            cout << "Add new keyframe id: " << curF.id_ << "\n";
            t11 = chrono::steady_clock::now();
            interaction->visualLastKF = *win.back();

        } else {
            // delete curF; // 释放非KF内存
            usleep(10 * 1000);
        }

        lastLastF = lastF;
        lastF = curF;
        interaction->trajectory.push_back({curF.Twc_.t_wb_});

        cout << "ExtractEdge spend: "
             << chrono::duration<double>(t2 - t1).count() << "s" << endl
             << "GenerateDTandDerivative spend: "
             << chrono::duration<double>(t3 - t2).count() << "s" << endl
             << "GenerateKeyPoint spend: "
             << chrono::duration<double>(t4 - t3).count() << "s" << endl
             << "UpdateDepth spend: "
             << chrono::duration<double>(t6 - t5).count() << "s" << endl
             << "CullingBadDepth spend: "
             << chrono::duration<double>(t7 - t6).count() << "s" << endl
             << "FuseDepth spend: " << chrono::duration<double>(t9 - t8).count()
             << "s" << endl
             << "AddOneKeyFeame transform spend: "
             << chrono::duration<double>(t11 - t10).count() << "s\n\n"
             << endl;

        while (interaction->stepBystep) {
            // 当前循环跑完，不需要再修改i
            usleep(100 * 1000);
        }
    }

#if defined(WRITE_MATCH_PAIR_IMAGE)
    if (KeyFrame::debugVideoWriter.isOpened()) {
        KeyFrame::debugVideoWriter.release();
    }
    if (KeyFrame::debugTriangulateWriter.isOpened()) {
        KeyFrame::debugTriangulateWriter.release();
    }
#endif

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
