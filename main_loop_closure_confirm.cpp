#include <unistd.h>


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
    string dataDir = "/home/laijinxiang/edge-slam/bdj3-record_data-i";
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
    const int getImgNum = 5;
    FindImageAndPose(firstImgIdx, vstrImages, vTimeStamps, vPriorPose, calib, imgs, vTwc, getImgNum-1);

    vector<Mat> im3;
    vector<Pose> vTwc3;
    FindImageAndPose(loopClosureImgIdx, vstrImages, vTimeStamps, vPriorPose, calib, im3, vTwc3, 1);
    imgs.push_back(im3[0]);
    vTwc.push_back(vTwc3[0]);

    // 以首帧作为参考帧
    for(int i = 1; i < vTwc.size(); ++i) {
        vTwc[i] = vTwc[0].Inverse() * vTwc[i];
    }
    vTwc[0] = Pose();

    shared_ptr<Camera> cam = make_shared<Camera>(kImageScale);

    chrono::steady_clock::time_point t0 = chrono::steady_clock::now();
    // 初始化关键帧
    vector<KeyFrame> kfs;
    for(int i = 0; i < vTwc.size(); ++i) {
        kfs.push_back({imgs[i], vTwc[i], cam, 1});
        kfs[i].CannyEdgeDetect();
        kfs[i].GenerateDTandDerivative();
        cout << "vTwc[" << i << "]: " << vTwc[i] << endl;
        ShowImage(kfs[i].edgeImg_[0], "edgeImg"+to_string(i), showImg);
    }

    // 头两帧进行地图点生成
    vector<vector<Eigen::Vector2d> > debugGoodKp1, debugGoodKp2;
    const int equalparts = 5;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    const size_t landmarkNum = kfs[0].GenerateLandmark(kfs[1], debugGoodKp1, debugGoodKp2, equalparts);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    cout << "landmarkNum : " << landmarkNum << endl;
    // for(int i = 0; i < equalparts; ++i) {
    //     DrawMatch(kfs[0].edgeImg_[0], kfs[1].edgeImg_[0], debugGoodKp1[i], debugGoodKp2[i], 
    //         "Epipolar constraint matches "+to_string(i), 1, 20);
    // }

    for(int i = 1; i < kfs.size() - 1; ++i) {
        kfs[0].UpdateDepth(kfs[i]);
    }

    auto it = kfs[0].landmark_.begin();
    while(it!=kfs[0].landmark_.end()) {
        if(!it->Converge()) {
            it = kfs[0].landmark_.erase(it);
            continue;
        }
        ++it;
    }
    cout << "kfs[0].landmark_.size: " << kfs[0].landmark_.size() << endl;

    vector<Landmark> noOptLandmark = kfs[0].landmark_;

    // 调用非线性优化进行BA
    KeyFrame &kf = kfs.back();
    vector<Mat> vDist;
    vector<Mat> vDx;
    vector<Mat> vDy;
    vector<Pose> T12;
    vector<Pose> T12_true;
    vector<Mat> edgeImg_true;

    for(int i = 1; i < getImgNum; ++i) {
        vDist.push_back(kfs[i].dist_[0]);
        vDx.push_back(kfs[i].dx_[0]);
        vDy.push_back(kfs[i].dy_[0]);
        T12.push_back(kfs[i].Twc_);
        T12_true.push_back(kfs[i].Twc_);
        edgeImg_true.push_back(kfs[i].edgeImg_[0]);
    }

    Optimizer optimizer(vDist, vDx, vDy, 1, 100, useInvZ, false);

    chrono::steady_clock::time_point t3 = chrono::steady_clock::now();
    const double cost = optimizer.Optimize(kfs[0].landmark_, T12);
    chrono::steady_clock::time_point t4 = chrono::steady_clock::now();

    cout << "\n\nprepare data spend: " << chrono::duration<double>(t1 - t0).count() << "s" << endl;
    cout << "generate landmark spend " << chrono::duration<double>(t2 - t1).count() << "s" << endl;
    cout << "optimize total spend " << chrono::duration<double>(t4 - t3).count() << "s" << endl;
    cout << "Pose diff: " << T12_true[0].Inverse() * T12[0] << endl;
    cout << "cost | mean: " << cost << " | " << cost/landmarkNum << endl;

    // 查看优化后的结果
    const int binWidth = kfs[0].grayImg_.cols / equalparts;

    vector<vector<Eigen::Vector2d> > projPx1(equalparts);
    vector<vector<vector<Eigen::Vector2d> > > reprojPx3(T12.size(), vector<vector<Eigen::Vector2d> >(equalparts));
    for(const Landmark &p : kfs[0].landmark_) {
        projPx1[p.uv_.x()/binWidth].push_back(p.uv_);

        for(int i = 0; i < T12.size(); ++i) {
            const Pose Tc2c1 = T12[i].Inverse();
            Eigen::Vector3d pc2 = Tc2c1 * p.GetPw();
            reprojPx3[i][p.uv_.x()/binWidth].push_back(cam->Project2PixelPlane(pc2));
        }
        // DrawMatch(kfs[0].edgeImg_[0], kf.edgeImg_[0], {projPx1.back()}, {reprojPx3.back()}, "one edge matche after optimization", 1, 1);
    }
    cout << endl;

    for(int i = 0; i < T12.size(); ++i) {
        for(int j = 0; j < equalparts; ++j) {
        DrawMatch(kfs[0].edgeImg_[0], edgeImg_true[i], projPx1[j], reprojPx3[i][j], 
            "edge matches after optimization "+to_string(i), 1, 10);
        }
    }

    // ShowPointCloud(kfs[0].landmark_, kfs[0].grayImg_);
    ShowPointCloud( noOptLandmark, kfs[0].landmark_);
    ShowPointCloud( noOptLandmark, kfs[0].landmark_, 1.0);
    return 0;
}
