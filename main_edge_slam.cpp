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

viz::Viz3d window("Local Map Viewer"); 
cv::Affine3d viewPose;

void ShowLocalMap(const set<Landmark* > &ps, const vector<Pose> &vTwc);
void UpdatePointCloud(vector<KeyFrame*> *historicalKF);
void Run(vector<KeyFrame*> *historicalKF);

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
        Assert(vPriorPose[i][0] > vPriorPose[i-1][0], "Check odom timestamp error!!!");
        // cout << fixed << vPriorPose[i][0] << " | " << vPriorPose[i-1][0] << endl;
    }

    WheelCameraCalib calib(config->Qcg, config->Pcg, config->wheelRadius);
    shared_ptr<Camera> cam = make_shared<Camera>(config);
    Optimizer optimizer(cam);
    
    KeyFrame *initFrame = nullptr;
    KeyFrame *lastKF = nullptr;
    KeyFrame *curKF = nullptr;
    thread *viewerThread;
    bool isInitialized = false;
    for(int i = firstImgIdx; i < vTimeStamps.size(); ++i) {
        Mat img;
        Pose Twc;
        GetImageAndPose(i, vstrImages, vTimeStamps, vPriorPose, calib, img, Twc);
        curKF = new KeyFrame(img, Twc, cam, 1);
        curKF->CannyEdgeDetect();
        curKF->GenerateDTandDerivative();
        if(!initFrame) {
            initFrame = curKF;
            // 首帧设置为单位矩阵
            initFrame->SetTwc(Pose());
            initFrame->InitializeLandmark();
            optimizer.AddOneKeyFeame(initFrame);
            //viewerThread = new thread(Run, &optimizer.historicalKF_);
            viewerThread = new thread(Run, &optimizer.window_);

            continue; // 认为初始化完毕
        }
        ShowImage(curKF->edgeImg_[0], "edgeImg"+to_string(i), showImg);
        
        // 使用KF更新当前帧的pose
        const Pose Twc1 = optimizer.window_.back()->priorTwc_;
        const Pose Twc2 = curKF->priorTwc_;
        const Pose Tc1c2 = Twc1.Inverse() * Twc2;
        curKF->SetTwc(optimizer.window_.back()->Twc_ * Tc1c2);
        
        // Step: 利用当前帧更新landmark depth，depth与host frame绑定
        const double convergeEdgeRatio = optimizer.window_.back()->UpdateDepth(*curKF);
        if(config->messageLevel <= MessageLevel::Error)
            cout << "convergeEdgeRatio: " << convergeEdgeRatio << endl;
        
        if(!isInitialized) {
            if(convergeEdgeRatio > 0.8) {
                // 初始化深度图已经生成，后续需要对每一帧进行深度图传播
                isInitialized = true;
            } else {
                continue;
            }   
        }

        // 利用生成的深度图，对当前帧进行位姿图优化，优化当前帧pose，同时将深度图传递给它
        // 将当前帧重投影点附近的深度值都赋值为基于高斯分布的深度
        // 在优化过程中，假设光度差服从t分布，可以计算出对应的优化权重值
        optimizer.SetInitLambda(1e-4);
        optimizer.UpdateCurrentFrame(curKF);
        double initDepthRatio = optimizer.TransformDepthMap2CurrentFrame(curKF);
        cout << "curKF depth map initialized depth ratio: " << initDepthRatio << endl;
        // ShowPointCloud(curKF->landmark_);
        // 当跟踪成功的点数量少于一定比例时，生成新的KF
        
        // Step: 当前帧选为新关键帧，
        // step1：追踪landmark，能够产生2D-2D的数据关联
        // step2：为剩余的edge point产生的landmark
        if(initDepthRatio < config->needNewKFMaxMatchEdgeRatio
            || NeedNewKF(optimizer.window_.back(), curKF) ) {   
            // 重叠度低，需要将当前帧选为KF，更新它的Landmark

            // 同时未跟踪上landmark的边缘点生成新的landmark
            curKF->InitializeLandmark();

            // 可视化步骤
            // optimizer.ShowLocalMap();

            optimizer.AddOneKeyFeame(curKF);
            if(optimizer.window_.size() > 2) {
                optimizer.SetInitLambda(1e-2);
                optimizer.SlidingWindowOptimize();
            }

        } else {
            delete curKF; // 释放非KF内存
            usleep(1 * 1000);
        }
    }

    viewerThread->join();
    delete viewerThread;

    return 0;
}

void ShowLocalMap(const set<Landmark* > &ps, const vector<Pose> &vTwc) {
    window.setViewerPose(viewPose);
    vector<Point3d> points;
    
    for(Landmark *p : ps) {
        if(p == nullptr || !p->Converge()) {
            continue;
        }
        const Eigen::Vector3d pw = p->GetPw();
        points.push_back({pw.x(), pw.y(), pw.z()});
    }
    vector<Vec3b> colors(points.size(), {0, 255, 0});

    viz::WCloud cloud(points, colors);
    // cloud.setColor(cv::viz::Color::green());
    // cloud.setSize(5);

    vector<Point3d> startEndCameraPos(2);
    for(int i = 0; i < vTwc.size(); ++i) {
        // Eigen默认列优先，这里先将其改为行优先以与Mat适配
        Eigen::Matrix<double, 4, 4, Eigen::RowMajor> _Twc = vTwc[i].ToMatrix4d();
        double *data = _Twc.data();
        cv::Mat mat44(4, 4, CV_64F, data);
        const cv::Affine3d Twc(mat44);
        
        if(i == 0 || i == vTwc.size()-1) {
            const Eigen::Vector3d t = vTwc[i].t_wb_;
            if(i == 0) {
                startEndCameraPos[0] = {t.x(), t.y(), t.z()};
            } else {
                startEndCameraPos[1] = {t.x(), t.y(), t.z()};
            }
        }
        // 显示坐标系
        //window.showWidget("cam"+to_string(i), viz::WCoordinateSystem(), Twc);
    }
 
    // 创建一个球体
    cv::viz::WSphere s0(startEndCameraPos[0], 0.01, 1, {255, 255, 255});
    cv::viz::WSphere s1(startEndCameraPos[1], 0.01, 1, {0, 255, 255});

    window.showWidget("PointCloud", cloud);
    bool shutdownViz = false;
    //window.showWidget("S0", s0);
    //window.showWidget("S1", s1);

    
    window.registerKeyboardCallback(ShutdownViz, &shutdownViz);
    // 运行事件循环，使窗口响应用户输入
    while (!shutdownViz) {
        window.spinOnce(1000);
    }
    // window.spin();
    window.removeAllWidgets();
    window.close();

    // 保留现场
    // viewPose = window.getViewerPose();
}

void UpdatePointCloud(vector<KeyFrame*> *historicalKF) {
    set<Landmark*> ps;
    vector<Pose> vTwc;

    vector<KeyFrame*> temp = *historicalKF;
    for(int i = 0; i < temp.size(); ++i) {
        // 新插入的最后一个KF未成熟
        KeyFrame *kf = temp[i];
        vTwc.push_back(kf->Twc_);
        for(Landmark *p : kf->landmark_) {
            if(p!=nullptr && !ps.count(p) && p->Converge()) {
                ps.insert(p);
            }
        }
    }
    if(!ps.empty()) {
        ShowLocalMap(ps, vTwc);
        cout << "show " << temp.size() << " KFs map points" << endl;
    } else {
        cerr << "wait for local map..." << endl;
    }
}


void Run(vector<KeyFrame*> *historicalKF) {
    while(1) {
        UpdatePointCloud(historicalKF);
        usleep(100 * 1000);
    }
}
