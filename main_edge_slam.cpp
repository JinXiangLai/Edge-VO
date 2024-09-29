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

// 利用极线约束去寻找anchor帧与普通帧的匹配以确定匹配特征点
// 得到一个较为准确的深度初值，再与闭环帧执行BA优化
// 结论，仅靠两帧生成的3D点存在很大的不确定性，而其为了剔除误匹配还是使用了描述子，
// 描述子都难以区分误匹配点，估计光度残差更难些

viz::Viz3d window("Local Map Viewer"); // 放在这里有问题
cv::Affine3d viewPose;

void ShowLocalMap(const set<Landmark* > &ps, const vector<Pose> &vTwc);
void Run(vector<KeyFrame*> *historicalKF);

int main(int argc, char** argv){

    // 读取程序参数
    string configFilePath = "../config.yaml";

    if (argc < 5){
        cerr << "[Error] Usage: ./main  useInverseDepth  showImage first_img_index loop_closure_img_index configFile" << endl;
        exit(-1);
    } else if(argc < 6) {
        cerr << "[Warning] Usage: ./main  useInverseDepth  showImage first_img_index loop_closure_img_index configFile" << endl;
        cout << "Default config: " << configFilePath << endl;
    }  else {
        configFilePath = string (argv[5]);
    }
    
    const bool useInvZ = bool (stoi(argv[1]));
    const bool showImg = bool(stoi(argv[2]));
    const int firstImgIdx = int(stoi(argv[3]));
    const int loopClosureImgIdx = int(stoi(argv[4]));
    Config _config(configFilePath);
    config = &_config;

    // 读取外部数据
    vector<string> vstrImages;
    vector<double> vTimeStamps;
    // TODO:需要将轮速系转换为相机系，所以倒不如直接在ORBSLAM3下的框架进行开发呢！！！
    vector<Eigen::Matrix<double, 8, 1>> vPriorPose;
    LoadImages(config->dataDir, vstrImages, vTimeStamps);
    LoadPriorOdom(config->dataDir, vPriorPose);

    WheelCameraCalib calib(config->Qcg, config->Pcg, config->wheelRadius);
    shared_ptr<Camera> cam = make_shared<Camera>(config);
    Optimizer optimizer(cam);
    
    KeyFrame *initFrame = nullptr;
    KeyFrame *lastKF = nullptr;
    KeyFrame *curKF = nullptr;
    thread *viewerThread;
    for(int i = firstImgIdx; i < vTimeStamps.size(); ++i) {
        Mat img;
        Pose Twc;
        GetImageAndPose(i, vstrImages, vTimeStamps, vPriorPose, calib, img, Twc);
        curKF = new KeyFrame(img, Twc, cam, 1);
        curKF->CannyEdgeDetect();
        curKF->GenerateDTandDerivative();
        if(!initFrame) {
            initFrame = curKF;
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
        const double recoverRatio = optimizer.window_.back()->UpdateDepth(*curKF);
        cout << "recoverRatio: " << recoverRatio << endl;
        
        // Step: 当前帧选为新关键帧，
        // step1：追踪landmark，能够产生2D-2D的数据关联
        // step2：为剩余的edge point产生的landmark
        if(recoverRatio < config->needNewKFMaxMatchEdgeRatio 
            || NeedNewKF(optimizer.window_.back(), curKF) ) {   
            // 重叠度低，需要将当前帧选为KF，更新它的Landmark
            const int reuseLandmarkNum = curKF->ReuseLandmark(optimizer.window_.back());
            cout << "reuseLandmarkNum: " << reuseLandmarkNum << endl;
            // 同时未跟踪上landmark的边缘点生成新的landmark
            curKF->InitializeLandmark();

            // 可视化步骤
            // optimizer.ShowLocalMap();

            optimizer.AddOneKeyFeame(curKF);
            optimizer.SlidingWindowOptimize();

            // ShowPointCloud(optimizer.window_.back()->landmark_);
        } else {
            delete curKF; // 释放非KF内存
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
        window.showWidget("cam"+to_string(i), viz::WCoordinateSystem(), Twc);
    }
 
    // 创建一个球体
    cv::viz::WSphere s0(startEndCameraPos[0], 0.1, 10, {255, 255, 255});
    cv::viz::WSphere s1(startEndCameraPos[1], 0.1, 10, {0, 255, 255});

    window.showWidget("PointCloud", cloud);
    window.showWidget("S0", s0);
    window.showWidget("S1", s1);

    // 运行事件循环，使窗口响应用户输入
    // window.spinOnce(1);
    window.spin();
    // window.close();

    // 保留现场
     window.removeAllWidgets();
    //window.removeWidget("PointCloud");
    // viewPose = window.getViewerPose();
}

void Run(vector<KeyFrame*> *historicalKF) {
    while(1) {
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
            cout << "show " << temp.size() << " KFs map points" << endl;;
        } else {
            // cout << "wait for local map..." << endl;
        }
        usleep(100 * 1000);
    }
}
