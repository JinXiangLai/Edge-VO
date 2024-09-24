#include <memory>
#include <unistd.h>


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

int main(int argc, char** argv){

    // 读取程序参数
    string dataDir = "/home/laijinxiang/docker-0105/dataset/0524-test-18/bdj3-record_data-i";
    if (argc < 5){
        cerr << "[Error] Usage: ./main  useInverseDepth  showImage first_img_index loop_closure_img_index [data directory]" << endl;
        exit(-1);
    } else if(argc < 6) {
        cerr << "[Warning] Usage: ./main  useInverseDepth  showImage first_img_index loop_closure_img_index [data directory]" << endl;
        cout << "Default dataDir: " << dataDir << endl;
    }  else {
        dataDir = string (argv[5]);
    }
    const bool useInvZ = bool (stoi(argv[1]));
    const bool showImg = bool(stoi(argv[2]));
    const int firstImgIdx = int(stoi(argv[3]));
    const int loopClosureImgIdx = int(stoi(argv[4]));

    // 读取外部数据
    vector<string> vstrImages;
    vector<double> vTimeStamps;
    // TODO:需要将轮速系转换为相机系，所以倒不如直接在ORBSLAM3下的框架进行开发呢！！！
    vector<Eigen::Matrix<double, 8, 1>> vPriorPose;
    LoadImages(dataDir, vstrImages, vTimeStamps);
    LoadPriorOdom(dataDir, vPriorPose);

    WheelCameraCalib calib;
    shared_ptr<Camera> cam = make_shared<Camera>(kImageScale);
    Optimizer optimizer(cam);
    
    vector<KeyFrame*> kfs;
    KeyFrame *initFrame = nullptr;
    KeyFrame *lastKF = nullptr;
    KeyFrame *curKF = nullptr;
    for(int i = firstImgIdx; i < vTimeStamps.size(); ++i) {
        Mat img;
        Pose Twc;
        GetImageAndPose(i, vstrImages, vTimeStamps, vPriorPose, calib, img, Twc);
        curKF = new KeyFrame({img, Twc, cam, 1});
        curKF->CannyEdgeDetect();
        curKF->GenerateDTandDerivative();
        if(!initFrame) {
            initFrame = curKF;
            kfs.push_back(initFrame);
            initFrame->InitializeLandmark();
            continue; // 认为初始化完毕
        }
        ShowImage(curKF->edgeImg_[0], "edgeImg"+to_string(i), showImg);
        
        
        // Step: 利用当前帧更新landmark depth，depth与host frame绑定
        const double recoverRatio = kfs.back()->UpdateDepth(*curKF);
        cout << "recoverRatio: " << recoverRatio << endl;
        
        // Step: 当前帧选为新关键帧，
        // step1：追踪landmark，能够产生2D-2D的数据关联
        // step2：为剩余的edge point产生的landmark
        if(recoverRatio < kNewKFMinMatchEdgeRatio || NeedNewKF(kfs.back(), curKF)) {   
            // 重叠度低，需要将当前帧选为KF，更新它的Landmark
            const int reuseLandmarkNum = curKF->ReuseLandmark(kfs.back());
            cout << "reuseLandmarkNum: " << reuseLandmarkNum << endl;
            // 同时未跟踪上landmark的边缘点生成新的landmark
            curKF->InitializeLandmark();
            kfs.push_back(curKF);

            optimizer.AddOneKeyFeame(curKF);

            // 可视化步骤
            optimizer.ShowLocalMap();
        } else {
            delete curKF; // 释放非KF内存
        }
    }

    return 0;
}
