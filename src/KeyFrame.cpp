#include <chrono>
#include <cstddef>
#include <cstdint>
#include <opencv2/highgui.hpp>
#include <string>
#include <vector>

#include "KeyFrame.h"
#include "Camera.h"
#include "Config.h"
#include "Eigen/src/Core/Matrix.h"
#include "Pose.h"
#include "Utils.h"

using namespace std;
using namespace cv;

class Landmark;

void KeyFrame::CannyEdgeDetect() {
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    cv::Mat blurred;
    // 应用高斯滤波来平滑边缘
    cv::GaussianBlur(grayImg_, blurred, cv::Size(5, 5), 1);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();

    double lowerThreshold = max(40.0, 60 * kImageScale); // 下限阈值
    double upperThreshold = max(60.0, 90 * kImageScale); // 上限阈值，越小提取边缘越多
    int apertureSize = 3;        // 应用Sobel算子的窗口大小
    Canny(blurred, edgeImg_[0], lowerThreshold, upperThreshold, apertureSize);
    chrono::steady_clock::time_point t3 = chrono::steady_clock::now();
    cout << "Extract canny edge spend " << chrono::duration<double>(t3 - t2).count() << "s" 
              << " & Gaussian Blur spend " << chrono::duration<double>(t2 - t1).count() << endl;

    vector<Point2i> px;
    // 取出边缘像素点
    for(int x = 0; x < edgeImg_[0].cols; ++x) {
        for(int y = 0; y < edgeImg_[0].rows; ++y) {
            if(edgeImg_[0].at<uchar>(y, x) == 255) {
                px.push_back({x, y});
            }
        }
    }

    chrono::steady_clock::time_point t4 = chrono::steady_clock::now();
    unPx_[0] = cam_->UndistortPoints(px); // 去畸变后的像素平面上的点
    chrono::steady_clock::time_point t5 = chrono::steady_clock::now();
    cout << "Undistort " << px.size() << "points spend " << chrono::duration<double>(t5 - t4).count() << "s" << endl;

    edgeImg_[0] = Mat::ones(edgeImg_[0].rows, edgeImg_[0].cols, CV_8UC1) * 255;
    vector<Eigen::Vector2d>::iterator it = unPx_[0].begin();
    int id = -1;
    while(it != unPx_[0].end()) {
        ++id;
        const Eigen::Vector2d &p = *it;
        // 边缘点置为黑色
        if(InRange(edgeImg_[0], {p.x(), p.y()}) ) {
            edgeImg_[0].at<uchar>(int(p.y()), int(p.x()) ) = 0;
            
            // 使用未去畸变像素邻域
            const int x = px[id].x, y = px[id].y;
            
            // 计算描述子
            // const Mat &m = grayImg_;
            // Eigen::Matrix<float, kDescriptorPatchSize, 1> d;
            // d << m.at<uchar>(y-1, x-1), m.at<uchar>(y-1, x), m.at<uchar>(y-1, x+1),
            //      m.at<uchar>(y, x-1), m.at<uchar>(y, x), m.at<uchar>(y, x+1), 
            //      m.at<uchar>(y+1, x-1), m.at<uchar>(y+1, x), m.at<uchar>(y+1, x+1);
            // descriptor_.emplace_back(d);

            descriptor_.push_back(CalculateDescriptor(grayImg_, {x, y}) );
            pointMapId_.insert({ {p.x(), p.y()}, descriptor_.size()-1});
            ++it;
        } else {
            it = unPx_[0].erase(it);
        }
    }
}

void KeyFrame::GenerateDTandDerivative() {
    dist_[0] = GetDistanceTransform(edgeImg_[0]);
    CaculateDerivative(dist_[0], dx_[0], dy_[0]);
}

size_t KeyFrame::GenerateLandmark(KeyFrame &kf2, vector<vector<Eigen::Vector2d> > &debugGoodKp1, vector<vector<Eigen::Vector2d> >&debugGoodKp2,
    const int equalparts) {
    
    const Pose T21 = kf2.Twc_.Inverse() * Twc_;
    const Pose T12 = T21.Inverse();
    cout << "T21: " << T21 << endl;
    cout << "T12: " << T12 << endl;
    debugGoodKp1.resize(equalparts);
    debugGoodKp2.resize(equalparts);
    for(int i = 0; i < unPx_[0].size(); ++i) {
        const Eigen::Vector2d &upx = unPx_[0][i];
        landmark_.push_back(new Landmark(upx, this, cam_, descriptor_[i], 1.0) );
        landmark_.back()->descriptor_ = descriptor_[i];
        // landmark_.back()->UpdateUncertainty();
    }

    return landmark_.size();
}

size_t KeyFrame::InitializeLandmark() {
    if(landmark_.empty() ){
        // initialKF会有该种情况
        landmark_.resize(unPx_[0].size(), nullptr);
    }
    
    for(int i = 0; i < unPx_[0].size(); ++i) {
        if(landmark_[i] != nullptr) {
            continue;
        }
        // shared_ptr<KeyFrame>(this)会导致多源智能指针，它会释放多次KeyFrame导致报错
        // 若需要使用智能指针，必须保证this在此前已经由一个智能指针管理，然后使用shared_from_this()来获取，否则只能使用原始指针
        // landmark_.push_back(make_shared<Landmark>(upx, make_shared<KeyFrame>(this), cam_, 1.0) ); [ERROR double free]
        landmark_[i] = new Landmark(unPx_[0][i], this, cam_, descriptor_[i], 1.0);
        // host帧也要增加与landmark的相互观测
        landmark_[i]->target_.insert({this, unPx_[0][i]});
    }

    return landmark_.size();
}

double KeyFrame::UpdateDepth(const KeyFrame &kf2) {
    double matchEdgeNum = 0;
    double convergeEdgeNum = 0; // 有效边缘点才参与统计重叠度

    for(int i = 0; i < landmark_.size(); ++i) {
        if(landmark_[i] == nullptr || landmark_[i]->IsOutOfRange()) {
            continue;
        }
        Landmark *pc1 = landmark_[i];
        if(pc1->Converge()) {
            convergeEdgeNum += 1.0;
        }
        // TODO: landmark会被其他帧观测到，所以不能一直使用host帧的像素进行深度更新?
        const vector<Eigen::Vector2d> kp2 = pc1->FindMatches(kf2);
        
        // 更新的是host帧下的深度
        const Pose T21 = kf2.Twc_.Inverse() * pc1->host_->Twc_;
        // cout << "T12: " << T21.Inverse() << endl;
        if(UpdateLandmarkDepth(kp2, T21, *cam_, *pc1) ) {
            // cout << "depth range, depth, std: [" << pc1.depthRange_[0] << " " << pc1.depthRange_[1] << "] " << pc1.z_ 
            //     << " " << pc1.uncertainty_ << endl;
            // DrawMatch(edgeImg_[0], kf2.edgeImg_[0], {pc1.uv_}, kp2, "current point 2 all Epipolar constraint matches", 1, 1);
            if(pc1->Converge() ) {
                matchEdgeNum += 1.0;
            }
        }   
    }
    
    if(convergeEdgeNum < landmark_.size() * 0.2) {
        // 有效路标点数量过低，需要继续进行深度滤波
        return 1.;
    }
    return matchEdgeNum / convergeEdgeNum;
}

int KeyFrame::ReuseLandmark(KeyFrame *kf1) {
    // 给新的KF2预分配内存
    landmark_ = vector<Landmark*>(unPx_[0].size(), nullptr);
    
    const int debugBin = 5;
    vector<vector<Eigen::Vector2d> > debugProj1(5), debugProj2(5);
    const int debugPart = edgeImg_[0].cols / 5; // 显示分区 

    int reuseLandmarkNum = 0;
    const vector<Landmark*> &landmark = kf1->landmark_;
    for(int i = 0; i < landmark.size(); ++i) {
        if(landmark[i] == nullptr || landmark[i]->IsOutOfRange()) {
            continue;
        }
        Landmark *pc1 = landmark[i];

        // 更新的是host帧下的depth
        const Eigen::Vector3d pc2 = Twc_.Inverse() * pc1->GetPw();
        if(pc2.z() < kMinDepth || pc2.z() > kMaxDepth) {
            continue;
        }

        const Eigen::Vector2i px2 = cam_->Project2PixelPlane(pc2).cast<int>();
        if(pointMapId_.count(px2) ) {
            const int vecId = pointMapId_[px2];
            if(landmark_[vecId] != nullptr) {
                // TODO: 选一个更好的，或者按照先来后到
                continue;
            }
            const uint64_t d2 = descriptor_[vecId];
            const uint64_t d1 = pc1->descriptor_;
            uint64_t score = CalculateDescriptorScore(d1, d2);
            if(score < kGoodDescriptorDist) {
                // 增加相互观测
                landmark_[vecId] = pc1;
                pc1->target_.insert({this, px2.cast<double>()});

                const int debugId = pc1->uv_.x()/debugPart;
                debugProj1[debugId].push_back(pc1->uv_);
                debugProj2[debugId].push_back(unPx_[0][vecId]);

                ++reuseLandmarkNum;
            }
        }

    }
    for(int i = 0; i < debugBin; ++i) {
        const string name("reuse landmark match part"); // +to_string(i));
        cv::namedWindow(name);
        // DrawMatch(kf1->edgeImg_[0], edgeImg_[0], debugProj1[i], debugProj2[i], 
        //     name, 1, 1);
    }
    // cv::destroyAllWindows();
    return reuseLandmarkNum;
}
