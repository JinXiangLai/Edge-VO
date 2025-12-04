#include <unistd.h>
#include <fstream>
#include <memory>
#include <random>  // 随机数引擎
#include <stack>
#include <thread>

#include <opencv2/viz/vizcore.hpp>

#include "Config.h"
#include "Initializer.h"
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
    if (!interaction->SetImgSaveFolderPath(
            fmt::format("/home/ht/Pictures/viz3d_screenshot",
                        config->debugMessageSaveFolder))) {
        cout << "Create viz3d image folder failed!\n";
    }
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
    Initializer initer(cam, &optimizer);

    KeyFrame lastF, lastLastF;

    thread* viewerThread = nullptr;
    thread* runWindowBAthread = nullptr;

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

        KeyFrame curF(img, Twc, cam, i, vTimeStamps[i], config->pyrLevel);
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
            if (!runWindowBAthread) {
                runWindowBAthread =
                    new thread(&Optimizer::RunWindowBA, &optimizer);
            }
            KeyFrame::InitPoseFileMessage();
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
            if (findMatchRatio > 0.95) {
                continue;
            }
            if (initer.InitializeSecondKeyFramePose(findMatchNum,
                                                    findMatchRatio, curF)) {

                int startRow = 0, stepRow = 30;
                initer.PutText2DebugMatchImg(fmt::format("t_12: {:.4f}", trans),
                                             startRow + stepRow);

                initer.WriteDebugInitImage();
                const string savePath =
                    fmt::format("{}/kf_id_{}_f_id_{}_t12_{:.3f}.jpeg",
                                config->debugMessageSaveFolder, win.back()->id_,
                                curF.id_, trans);
                cv::imwrite(savePath, initer.debugMatchImg_);

                optimizer.AddOneKeyFeame(new KeyFrame(curF));
                interaction->visualCurF = curF;  // 记录优化pose后的当前帧
                // 初始化深度图已经生成，后续需要对每一帧进行深度图传播
                isInitialized = true;
                cout << "\n******\nInitialized!\n******\n";
                cv::imshow("init 2 KF", initer.debugMatchImg_);
                cv::waitKey();
                cv::destroyWindow("init 2 KF");
                // 模型预热
                usleep(1000 * 1e3);
                while (optimizer.newKF_ != nullptr) {
                    usleep(100 * 1e3);
                }
            } else if (findMatchRatio < 0.7 || findMatchNum < 200) {
                cout << "Few match to initialize! Reset!" << endl;
                ResetStatus(&optimizer, &isInitialized, &trackLostCount);
            }

            continue;
        }

        if (config->debugRunSerially) {
            while (optimizer.newKF_ != nullptr) {
                usleep(1 * 1000);
            }
        }
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
            (findMatchRatio < 0.7 || findMatchNum < 200);

        const Pose T12 = win.back()->Tcw_ * curF.Twc_;
        const double horDist = T12.t_wb_.head(2).norm();
        const double meanDepth = optimizer.GetLastKFmeanDepth();
        const bool caseMoveBaselineLOng =
            meanDepth > 0 && horDist > meanDepth * 0.5;

        // 未全部完成初始化时，快速插入KF
        const bool caseFastInsertKF =
            findMatchRatio < 0.9 &&
            static_cast<int>(win.size()) < config->maxKFnumInWindow;

        const bool hasHorMove = (horDist > 0.05 * meanDepth);

        const double frameDuration =
            (curF.timestamp_ - win.back()->timestamp_) * 1e3;  // ms
        const bool longTimeNoInsertKF =
            frameDuration > 2.0 * optimizer.lastWinBAspendTime_ &&
            findMatchRatio < 0.8;

        // 必须保证当前KF收敛足够多的点了
        cout << fmt::format(
            "Need KF check: findMatchRatio:{:.1f}, findMatchNum: {}, "
            "trans "
            "dist: {:.2f}, "
            "rot ang: {:.1f}deg\n",
            findMatchRatio, findMatchNum, T12.t_wb_.head(2).norm(),
            Quat2RPY(T12.q_wb_).norm() * kRad2Deg);

        // 检验地图点跟踪效果，光流跟踪效果和运行基线
        if (optimizer.CanAddNewKF() &&
            ((caseOptflowTrackLow && hasHorMove) || caseMoveBaselineLOng ||
             caseFastInsertKF || trackLocalMapLow || longTimeNoInsertKF)) {
            cout << fmt::format(
                "add kf case: trackLocalMapLow: {}, caseOptflowTrackLow: {}, "
                "caseMoveBaselineLOng: {}, caseFastInsertKF: {}, "
                "longTimeNoInsertKF: {}\n",
                trackLocalMapLow, caseOptflowTrackLow, caseMoveBaselineLOng,
                caseFastInsertKF, longTimeNoInsertKF);

            optimizer.AddOneKeyFeame(new KeyFrame(curF));
            // TODO：当前帧被选为关键帧时，需要进行多帧的局部BA优化，因此需要添加互观测
            cout << "Add new keyframe id: " << curF.id_ << "\n";
            interaction->visualLastKF = *win.back();
        }
        interaction->trajectory.push_back({curF.Twc_.t_wb_});

        const double frameTimeGap =
            (curF.timestamp_ - lastF.timestamp_) * 1e3;  // ms

        cout << fmt::format(
            "TrackWithOpticalFlow spend:{:.3f}ms, TrackLocalMap spend: "
            "{:.3f}ms\n\n",
            ChronoMillisecTimeDuration(t1, t2),
            ChronoMillisecTimeDuration(t2, t3));
        chrono::steady_clock::time_point t4 = chrono::steady_clock::now();
        const double trackSpendTime = ChronoMillisecTimeDuration(t1, t4);
        const double sleepTime = min((frameTimeGap - trackSpendTime), 28.0);
        if (sleepTime > 0. &&
            (config->debugShowOnlineResult3D || !config->debugRunSerially)) {
            usleep(sleepTime * 1e3);
        } else {
            cout << fmt::format(
                "Error tracking spend too much time! curF.timestamp_: {}s, "
                "lastF.timestamp_: {}s, frameTimeGap: {:.1f}ms, "
                "trackSpendTime: {:.1f}ms, sleepTime: {:.1f}.\n",
                curF.timestamp_, lastF.timestamp_, frameTimeGap, trackSpendTime,
                sleepTime);
        }
        if (i % 30 == 0) {
            cout << fmt::format(
                "Tracking spend time: curF.timestamp_: {}s, "
                "lastF.timestamp_: {}s, frameTimeGap: {:.1f}ms, "
                "trackSpendTime: {:.1f}ms, sleepTime: {:.1f}.\n",
                curF.timestamp_, lastF.timestamp_, frameTimeGap, trackSpendTime,
                sleepTime);
        }

        KeyFrame::WritePoseMessage2File(curF);
        lastLastF = lastF;
        lastF = curF;

        while (interaction->stepBystep) {
            // 当前循环跑完，不需要再修改i
            usleep(100 * 1000);
        }
    }

    const chrono::steady_clock::time_point& tEnd = chrono::steady_clock::now();
    cout << fmt::format(
                "process {} images of sequence {}, total spend:{:.3f}s, "
                "sequence "
                "record duration: {:.3f}s",
                vTimeStamps.size() - firstImgIdx,
                config->dataDir.substr(config->dataDir.find_last_of('/') + 1),
                ChronoMillisecTimeDuration(tStart, tEnd) * 1e-3,
                vTimeStamps.back() - vTimeStamps[firstImgIdx])
         << endl;

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

    if (initer.debugInitTrackWriter_.isOpened()) {
        initer.debugInitTrackWriter_.release();
    }

    // 停止后端优化线程
    optimizer.StopRunBA();
    runWindowBAthread->join();
    delete runWindowBAthread;
    cout << "Window BA thread recycled!" << endl;

    for (const KeyFrame* kf : win) {
        KeyFrame::WritePoseMessage2File(*kf);
    }
    KeyFrame::ProcessPoseFile();

    // 调用ls命令，列出当前目录下的文件
    cout << "Run evo evaluate kf traj precision!";
    system(fmt::format("evo_ape tum {}/groundtruth.txt {} -a -s",
                       config->dataDir, KeyFrame::kfPoseFilePath)
               .c_str());

    if (config->debugShowOnlineResult3D) {
        viewerThread->join();
        delete viewerThread;
    }
    cout << "Viwe 3D thread recycled!" << endl;

    return 0;
}

void Run(Optimizer* optimizer) {
    while (!interaction->stopView) {
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
