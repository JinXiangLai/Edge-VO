#include <chrono>

#include "KeyFrame.h"
#include "Utils.h"

using namespace std;
using namespace cv;

class Landmark;

void KeyFrame::CannyEdgeDetect() {
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    cv::Mat blurred;
    // 应用高斯滤波来平滑边缘
    cv::GaussianBlur(grayImg_, blurred, cv::Size(3, 3), 1);
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
            pointMapId_.insert({tuple<int, int>(p.x(), p.y()), descriptor_.size()-1});
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
        // shared_ptr<KeyFrame>(this)会导致多源智能指针，它会释放多次KeyFrame导致报错
        landmark_.push_back(new Landmark(upx, this, cam_, 1.0) );
        // 使用make_shared无法直接创建指向同一个this的智能指针对象，所以最终还是得像ORBSLAM那样直接使用原始指针?
        // 若需要使用智能指针，必须保证this在此前已经由一个智能指针管理，然后使用shared_from_this()来获取，否则只能使用原始指针
        //landmark_.push_back(make_shared<Landmark>(upx, make_shared<KeyFrame>(this), cam_, 1.0) );
        // 给地图点赋值描述子
        landmark_.back()->descriptor_ = descriptor_[i];
        // landmark_.back()->UpdateUncertainty();
    }

    return landmark_.size();
}

void KeyFrame::UpdateDepth(const KeyFrame &kf2) {
    const Pose T21 = kf2.Twc_.Inverse() * Twc_;
    cout << "T12: " << T21.Inverse() << endl;
    for(int i = 0; i < landmark_.size(); ++i) {
        if(landmark_[i] == nullptr) {
            continue;
        }
        Landmark &pc1 = *landmark_[i];
        const vector<Eigen::Vector2d> kp2 = pc1.FindMatches(kf2);
        
        if(UpdateLandmarkDepth(kp2, T21, *cam_, pc1) ) {
            cout << "[" << pc1.depthRange_[0] << " " << pc1.depthRange_[1] << "] " << pc1.z_ << endl;
            // DrawMatch(edgeImg_[0], kf2.edgeImg_[0], {pc1.uv_}, kp2, "current point 2 all Epipolar constraint matches", 1, 1);
        }   
    }
}
