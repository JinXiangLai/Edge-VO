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

const cv::Point kViz3DWindowPos(1920 + 1920 / 2, 1080 / 2);

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
    // window.setWindowPosition(kViz3DWindowPos);
    interaction->window = &window;
    InitColor();

    const bool useInvZ = config->useInvZ;
    const bool showImg = config->showDebugImg;
    const int firstImgIdx = config->firstImgIdx;
    const int loopClosureImgIdx = config->loopClosureImgIdx;

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

    for (int i = 1; i < vTimeStamps.size(); ++i) {
        Assert(vTimeStamps[i] - vTimeStamps[i - 1] > 0,
               "Check img timestamp error!!!");
    }
    for (int i = 1; i < vPriorPose.size(); ++i) {
        //cout << fixed << vPriorPose[i][0] << " | " << vPriorPose[i-1][0] << endl;
        Assert(vPriorPose[i][0] >= vPriorPose[i - 1][0],
               "Check odom timestamp error!!!");
    }

    WheelCameraCalib calib(config->Qcg, config->Pcg, config->wheelRadius);
    shared_ptr<Camera> cam = make_shared<Camera>(config);
    Optimizer optimizer(cam);

    KeyFrame* initFrame = nullptr;
    KeyFrame lastF, lastLastF;
#if defined(SHOW_ONLINE_3D_RESULT)
    thread* viewerThread;
#endif
    bool isInitialized = false;
    vector<KeyFrame*>& win = optimizer.window_;
    double accDist = 0.;
    stack<KeyFrame>
        unmappedFrame;  // 由于当前把每一帧都用来更新深度，所以不需要像lsd slam那样保留一些帧
    for (int i = firstImgIdx; i < vTimeStamps.size(); ++i) {
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

#if 1
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        curF.CannyEdgeDetect();
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        curF.GenerateDTandDerivative();
        chrono::steady_clock::time_point t3 = chrono::steady_clock::now();
        chrono::steady_clock::time_point t4 = chrono::steady_clock::now();
#else
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        curF.ExtractEdge();
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        curF.GenerateDTandDerivative();
        chrono::steady_clock::time_point t3 = chrono::steady_clock::now();
        curF.GenerateKeyPoint();
        chrono::steady_clock::time_point t4 = chrono::steady_clock::now();
#endif

        if (!initFrame) {
            initFrame = new KeyFrame(curF);
            // 首帧设置为单位矩阵
            initFrame->SetTwc(Pose());
            initFrame->depthImage_ = depthImg;
            lastF = *initFrame;
            initFrame->InitializeLandmark();
            optimizer.AddOneKeyFeame(initFrame);
            interaction->visualLastKF = win.back();
#if defined(SHOW_ONLINE_3D_RESULT)
            viewerThread = new thread(Run, &optimizer);
#endif
            continue;  // 认为初始化完毕
        }
        ShowImage(curF.edgeImg_[0], "edgeImg" + to_string(i), showImg);

        // 使用KF更新当前帧的pose
        const Pose Twc2 = curF.priorTwc_;
        Pose Twc1, Tc1c2;
        Twc1 = win.back()->priorTwc_;
        Tc1c2 = Twc1.Inverse() * Twc2;
        curF.SetTwc(win.back()->Twc_ * Tc1c2);

        const double trans =
            (lastF.priorTwc_.Inverse() * curF.priorTwc_).t_wb_.norm();
        accDist += trans;

        // 显示线程使用
        interaction->visualCurF = curF;
        interaction->visualCurFinit = curF;

        if (!isInitialized) {
            // 初始化深度图
            lastLastF = lastF;
            lastF = curF;
            const double kfConvergeEdgeRatio = win.back()->UpdateDepth(curF);
            if (config->messageLevel <= MessageLevel::Error)
                cout << "kfConvergeEdgeRatio, accDist: " << kfConvergeEdgeRatio
                     << ", " << accDist << endl;
            if (kfConvergeEdgeRatio > 0.1 ||
                (accDist > 0.1 && (curF.id_ - initFrame->id_ > 30)) ||
                accDist > config->needNewKFtrans || config->useDepthImage) {
                // 初始化深度图已经生成，后续需要对每一帧进行深度图传播
                isInitialized = true;
                accDist = 0.;
                cout << "\n******\nInitialized!\n******\n";
            } else {
                continue;
            }
        }

#if 1
        // Step: 利用当前帧更新深度图
        // step1: 优化当前帧pose
        optimizer.SetInitLambda(1.0);
        // // TODO: 图像存在运动模糊时，会导致landmark, pose估计出异常值，
        // // 导致sliding window optimization优化崩溃：可仅优化pose而不优化landmark
        bool needKFbySight = false;
        //optimizer.TrackLocalMap(
        //    &curF, needKFbySight);  // TODO: 问题是这里的pose估计不准

        // step2: 利用当前帧更新landmark depth，depth与host frame绑定
        chrono::steady_clock::time_point t5 = chrono::steady_clock::now();
        const double kfConvergeEdgeRatio = win.back()->UpdateDepth(curF);
        chrono::steady_clock::time_point t6 = chrono::steady_clock::now();
        // win.back()->CullingBadDepth(&curF);
        chrono::steady_clock::time_point t7 = chrono::steady_clock::now();

        chrono::steady_clock::time_point t8, t9;
        if (win.back()->updateFrameCount_ % 5 == 0 && accDist > 0.03) {
            chrono::steady_clock::time_point t8 = chrono::steady_clock::now();
            win.back()->FuseDepth();
            chrono::steady_clock::time_point t9 = chrono::steady_clock::now();
            //cout << "Fuse depth spend " << chrono::duration<double>(t2 - t1).count() << "s" << endl;
        }

#else
        // step1
        const double kfConvergeEdgeRatio = win.back()->UpdateDepth(curF);
        // step2
        optimizer.SetInitLambda(0);
        //optimizer.TrackLocalMap(&curF);
#endif

        // 显示线程更新使用
        interaction->visualCurF = curF;  // 记录优化pose后的当前帧
        // step3: 剔除地图外点, TODO: 应该使用融合而不是剔除策略！！！
        // optimizer.CullingErrorLandmark(&curF);

        // step4: 将深度图传递给当前帧
        // 将当前帧重投影点附近的深度值都赋值为基于高斯分布的深度
        // 在优化过程中，假设光度差服从t分布，可以计算出对应的优化权重值
        // double initDepthRatio = optimizer.TransformDepthMap2CurrentFrame(&curF);
        // cout << "curF depth map initialized depth ratio: " << initDepthRatio << endl;
        // win.back()->FuseDepth();

        // if((lastF.Twc_.Inverse() * curF.Twc_).t_wb_.norm() > 0.2) {
        //     ShowPointCloud(curF.landmark_);
        //     // 只能赋值内容，不能赋值地址
        //     lastF = curF;
        // }

        // 当跟踪成功的点数量少于一定比例且运动满足阈值时，生成新的KF
        // Step: 当前帧选为新关键帧，
        // step1：追踪landmark，能够产生2D-2D的数据关联
        // step2：为剩余的edge point产生的landmark
        const Pose T12 = win.back()->priorTwc_.Inverse() * curF.priorTwc_;
        bool case1 =
                 false,  // initDepthRatio < config->needNewKFMaxMatchEdgeRatio,
            case2 = true,  // kfConvergeEdgeRatio > 0.3,
            case3 = T12.t_wb_.norm() > config->needNewKFtrans,
             case4 =
                 Quat2RPY(T12.q_wb_).norm() * kRad2Deg > config->needNewKFrot,
             case5 = accDist > config->needNewKFtrans,
             case6 = curF.id_ - win.back()->id_ > 5;
        // 必须保证当前KF收敛足够多的点了
        cout << "case1-5: " << case1 << " " << case2 << " " << case3 << " "
             << case4 << " " << case5 << " accdist: " << accDist << endl;
        chrono::steady_clock::time_point t10, t11;
        if (((case1 || case3 || case4 || case5) && case2 && case6) ||
            needKFbySight) {
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

            accDist = 0.;

#ifndef USE_DT_RESIDUAL
            // 同时未跟踪上landmark的边缘点生成新的landmark
            curF.InitializeLandmark();
#else
            // nothing
#endif
            // 可视化滑窗内点云
            // ShowPointCloud(curF.landmark_);
            // optimizer.ShowLocalMap(nullptr);

            win.back()->FuseDepth();
            t10 = chrono::steady_clock::now();
            optimizer.AddOneKeyFeame(new KeyFrame(curF));
            t11 = chrono::steady_clock::now();
            interaction->visualLastKF = win.back();

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

#if defined(SHOW_ONLINE_3D_RESULT)
    viewerThread->join();
    delete viewerThread;
#endif

    return 0;
}

#define SHOW_GLOBAL_MAP 0
void Run(Optimizer* optimizer) {
    while (1) {
#if SHOW_GLOBAL_MAP
        interaction->ShowGlobalMapPoint();
#else
        // UpdatePointCloud(historicalKF);
        if (!optimizer->window_.empty()) {
            optimizer->ShowLocalMap();
        }
#endif
        usleep(10 * 1000);
    }
}
