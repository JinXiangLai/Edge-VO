#include <fstream>
#include <memory>
#include <thread>
#include <unistd.h>

#include <opencv2/viz/vizcore.hpp>

#include "Config.h"
#include "Landmark.h"
#include "Pose.h"
#include "Utils.h"
#include "Optimizer.h"
#include "WheelCameraCalib.h"

using namespace std;
using namespace cv;

void Run(Optimizer *optimizer);

int main(int argc, char** argv){

    // 读取程序参数
    //string configFilePath = "../config.yaml";
    string configFilePath = "../tum_config.yaml";


    if (argc < 2){
        cerr << "[WARNING] Usage: ./main configFile[DEFAULT: " << configFilePath << "]" << endl;
    } else {
        configFilePath = string (argv[1]);
        cerr << "[INFO] configFile: " << configFilePath << endl;
    }
    
    Config _config(configFilePath);
    config = &_config;
    InteractionParam _visualizeParam;
    interaction = &_visualizeParam;
    viz::Viz3d window("Local Map Viewer");
    interaction->window = &window;

    const bool useInvZ = config->useInvZ;
    const bool showImg = config->showDebugImg;
    const int firstImgIdx = config->firstImgIdx;
    const int loopClosureImgIdx = config->loopClosureImgIdx;


    // 读取外部数据
    vector<string> vstrImages;
    vector<double> vTimeStamps;
    // TODO:需要将轮速系转换为相机系，所以倒不如直接在ORBSLAM3下的框架进行开发呢！！！
    vector<Eigen::Matrix<double, 8, 1>> vPriorPose;
    if(config->model == "pinhole") {
        LoadImages(config->dataDir, vstrImages, vTimeStamps, ".png");
    } else {
        LoadImages(config->dataDir, vstrImages, vTimeStamps);
    }
    LoadPriorOdom(config->dataDir, vPriorPose);

    for(int i = 1; i < vTimeStamps.size(); ++i) {
        Assert(vTimeStamps[i]-vTimeStamps[i-1] > 0, "Check img timestamp error!!!");
    }
    for(int i = 1; i < vPriorPose.size(); ++i) {
        //cout << fixed << vPriorPose[i][0] << " | " << vPriorPose[i-1][0] << endl;
        Assert(vPriorPose[i][0] >= vPriorPose[i-1][0], "Check odom timestamp error!!!");
    }

    WheelCameraCalib calib(config->Qcg, config->Pcg, config->wheelRadius);
    shared_ptr<Camera> cam = make_shared<Camera>(config);
    Optimizer optimizer(cam);
    
    KeyFrame *initFrame = nullptr;
    KeyFrame lastF;
    thread *viewerThread;
    bool isInitialized = false;
    vector<KeyFrame *> &win = optimizer.window_;
    double accDist = 0.;
    vector<KeyFrame> unmappedFrame; // 新KF与过去的F匹配来产生新的深度估计
    for(int i = firstImgIdx; i < vTimeStamps.size(); ++i) {
        if(interaction->stepBystep) {
            usleep(100 * 1000);
            --i;
            continue;
        }
        Mat img;
        Pose Twc;
        GetImageAndPose(i, vstrImages, vTimeStamps, vPriorPose, calib, img, Twc);
        cout << i << " th cur img timestamp: " << to_string(vTimeStamps[i]) << endl;

        KeyFrame curF(img, Twc, cam, i, 1);
        curF.CannyEdgeDetect();
        curF.GenerateDTandDerivative();

        if(!initFrame) {
            initFrame = new KeyFrame(curF);
            lastF = curF;

            // 首帧设置为单位矩阵
            initFrame->SetTwc(Pose());
            initFrame->InitializeLandmark();
            optimizer.AddOneKeyFeame(initFrame);
            interaction->visualLastKF = win.back();
            viewerThread = new thread(Run, &optimizer);
            continue; // 认为初始化完毕
        }
        ShowImage(curF.edgeImg_[0], "edgeImg"+to_string(i), showImg);

        //cv::imshow("cur f"+to_string(vTimeStamps[i]), curF.grayImg_);
        //cv::waitKey(56);
        //cv::destroyWindow("cur f"+to_string(vTimeStamps[i]));
        
        // 使用KF更新当前帧的pose
        const Pose Twc1 = win.back()->priorTwc_;
        const Pose Twc2 = curF.priorTwc_;
        const Pose Tc1c2 = Twc1.Inverse() * Twc2;
        curF.SetTwc(win.back()->Twc_ * Tc1c2);
        // 显示线程使用
        interaction->visualCurF = curF;
        
        if(!isInitialized) {
            // 初始化深度图
            const double trans = (lastF.Twc_.Inverse() * curF.Twc_).t_wb_.norm();
            accDist += trans;
            lastF = curF;

            const double kfConvergeEdgeRatio = win.back()->UpdateDepth(curF);
            if(config->messageLevel <= MessageLevel::Error)
                cout << "kfConvergeEdgeRatio: " << kfConvergeEdgeRatio << endl;
            if(kfConvergeEdgeRatio > 0.6 || (accDist > 0.5 && (curF.id_ - initFrame->id_ > 30)) ) {
                // 初始化深度图已经生成，后续需要对每一帧进行深度图传播
                isInitialized = true;
                accDist = 0.;
                cout << "\n******\nInitialized!\n******\n";
            } else {
                continue;
            } 
        }

#if 0
        // Step: 利用当前帧更新深度图
        // step1: 优化当前帧pose
        optimizer.SetInitLambda(1e-3);
        // TODO: 图像存在运动模糊时，会导致landmark, pose估计出异常值，
        // 导致sliding window optimization优化崩溃：可仅优化pose而不优化landmark
        optimizer.TrackLocalMap(&curF);
        
        // step2: 利用当前帧更新landmark depth，depth与host frame绑定
        const double kfConvergeEdgeRatio = win.back()->UpdateDepth(curF);
#else
        // step1
        const double kfConvergeEdgeRatio = win.back()->UpdateDepth(curF);
        // step2
        optimizer.SetInitLambda(0);
        optimizer.TrackLocalMap(&curF);
#endif        
        // 显示线程更新使用
        interaction->visualCurF = curF;
        // step3: 剔除地图外点
        optimizer.CullingErrorLandmark(&curF);
        
        // step4: 将深度图传递给当前帧
        // 将当前帧重投影点附近的深度值都赋值为基于高斯分布的深度
        // 在优化过程中，假设光度差服从t分布，可以计算出对应的优化权重值
        double initDepthRatio = optimizer.TransformDepthMap2CurrentFrame(&curF);
        cout << "curF depth map initialized depth ratio: " << initDepthRatio << endl;
        
        const double trans = (lastF.Twc_.Inverse() * curF.Twc_).t_wb_.norm();
        accDist += trans;
        // if((lastF.Twc_.Inverse() * curF.Twc_).t_wb_.norm() > 0.2) {
        //     ShowPointCloud(curF.landmark_);
        //     // 只能赋值内容，不能赋值地址
        //     lastF = curF;
        // }
        lastF = curF;
        
        // 当跟踪成功的点数量少于一定比例且运动满足阈值时，生成新的KF
        // Step: 当前帧选为新关键帧，
        // step1：追踪landmark，能够产生2D-2D的数据关联
        // step2：为剩余的edge point产生的landmark
        const Pose T12 = win.back()->priorTwc_.Inverse() * curF.priorTwc_;
        bool case1 = initDepthRatio < config->needNewKFMaxMatchEdgeRatio,
             case2 = kfConvergeEdgeRatio > 0.3,
             case3 = T12.t_wb_.norm() > config->needNewKFtrans,
             case4 = Quat2RPY(T12.q_wb_).norm() * kRad2Deg > config->needNewKFrot,
             case5 = accDist > config->needNewKFtrans;
        // 必须保证当前KF收敛足够多的点了
        if((case1 || case3 || case4 || case5) && case2) {
            {
                static bool first = true;
                ofstream f;
                const string name = "generate_KF_case.csv";
                if(first) {
                    f.open(name.c_str(), ios::out);
                    first = false;
                    f.close();
                }
                f.open(name.c_str(), ios::app);
                f << "(" <<case1 << " || " << case3 << " || " << case4 << " || " << case5 << ") && " << case2 << endl;
                f.close();
                
            }
            // 重叠度低，需要将当前帧选为KF，更新它的Landmark
            if(curF.unPx_.size() < win.back()->unPx_.size() * 0.6) {
                // TODO：显示线程会显示出异常的图像，需要检测并剔除异常图像
                continue;
            } else {
                accDist = 0;
            }

#ifndef USE_DT_RESIDUAL      
            // 同时未跟踪上landmark的边缘点生成新的landmark
            curF.InitializeLandmark();
#else
            // nothing
#endif
            // 可视化滑窗内点云
            // ShowPointCloud(curF.landmark_);
            // optimizer.ShowLocalMap(nullptr);


            optimizer.AddOneKeyFeame(new KeyFrame(curF) );
            interaction->visualLastKF = win.back();

            if(win.size() > 2) {
                optimizer.SetInitLambda(1.0);
                optimizer.SlidingWindowOptimize();
                // optimizer.UpdateDepthInWindow();
            }
            // optimizer.CullingErrorLandmark();

        } else {
            // delete curF; // 释放非KF内存
            usleep(10 * 1000);
        }

        while(interaction->stepBystep) {
            // 当前循环跑完，不需要再修改i
            usleep(100 * 1000);
        }

    }

    viewerThread->join();
    delete viewerThread;

    return 0;
}

void Run(Optimizer *optimizer) {
    while(1) {
        // UpdatePointCloud(historicalKF);
        if(!optimizer->window_.empty()) {
            optimizer->ShowLocalMap();
        }
        usleep(10 * 1000);
    }
}
