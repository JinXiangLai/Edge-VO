#include <memory>
#include <unistd.h>


#include "Landmark.h"
#include "Utils.h"
#include "Optimizer.h"

using namespace std;
using namespace cv;

// 利用极线约束去寻找anchor帧与普通帧的匹配以确定匹配特征点
// 得到一个较为准确的深度初值，再与闭环帧执行BA优化
// 结论，仅靠两帧生成的3D点存在很大的不确定性，而其为了剔除误匹配还是使用了描述子，
// 描述子都难以区分误匹配点，估计光度残差更难些

int main(int argc, char** argv){

    // 读取程序参数
    string dataDir = "/home/laijinxiang/docker-0105/dataset/0524-test-18/bdj3-record_data-i";
    //string dataDir = "/home/laijinxiang/edge-slam/bdj3-record_data-i";

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
    // 找到3张图像及对应的Pose
    vector<Mat> imgs;
    vector<Pose> vTwc;
    WheelCameraCalib calib;
    const int getImgNum = 30;
    FindImageAndPose(firstImgIdx, vstrImages, vTimeStamps, vPriorPose, calib, imgs, vTwc, getImgNum);

    shared_ptr<Camera> cam = make_shared<Camera>(kImageScale);

    chrono::steady_clock::time_point t0 = chrono::steady_clock::now();
    // 初始化关键帧
    vector<KeyFrame*> kfs;
    for(int i = 0; i < vTwc.size(); ++i) {
        kfs.push_back(new KeyFrame(imgs[i], vTwc[i], cam, 1));
        kfs[i]->CannyEdgeDetect();
        kfs[i]->GenerateDTandDerivative();
        cout << "vTwc[" << i << "]: " << vTwc[i] << endl;
        ShowImage(kfs[i]->edgeImg_[0], "edgeImg"+to_string(i), showImg);
    }

    // 头两帧进行地图点生成
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    const size_t landmarkNum = kfs[0]->InitializeLandmark();
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    cout << "landmarkNum : " << landmarkNum << endl;
    Optimizer optimizer(cam);
    optimizer.AddOneKeyFeame(kfs[0]);

    const int maxUpdateId = kfs.size();
    for(int i = 1; i < maxUpdateId; ++i) {
        // kfs[0].UpdateDepth(kfs[i]);
        const double coverRatio = optimizer.window_.back()->UpdateDepth(*kfs[i]);
        // 产生新KF
        if(i%10 == 0) {
            const int reuseLandmarkNum = kfs[i]->ReuseLandmark(optimizer.window_.back());
            cout << i << " th reuseLandmarkNum: " << reuseLandmarkNum << endl;
            kfs[i]->InitializeLandmark();
            optimizer.AddOneKeyFeame(kfs[i]);
        }
    }

    cout << "kfs[0].landmark_.size: " << kfs[0]->landmark_.size() << endl;

    optimizer.SlidingWindowOptimize();

    return 0;
}
