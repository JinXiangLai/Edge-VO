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


vector<Eigen::Vector2d> KeyFrame::FindMatches(const Eigen::Vector2d &kp1, const Pose &Twc1) {
    const Pose T21 = Twc_.Inverse() * Twc1;
    // 需要全局函数作用符"::"以实现类外全局函数的调用
    Landmark pc1(kp1, make_shared<Pose>(Twc1), cam_, 1.0);
    const KeyFrame &kf2 = *this;
    vector<Eigen::Vector2d> kp2 = ::FindMatches(pc1, kf2, T21, *cam_);
    return kp2;
}

size_t KeyFrame::GenerateLandmark(KeyFrame &kf2, vector<vector<Eigen::Vector2d> > &debugGoodKp1, vector<vector<Eigen::Vector2d> >&debugGoodKp2,
    const int equalparts) {
    
    const Pose T21 = kf2.Twc_.Inverse() * Twc_;
    const Pose T12 = T21.Inverse();
    cout << "T21: " << T21 << endl;
    cout << "T12: " << T12 << endl;
    debugGoodKp1.resize(equalparts);
    debugGoodKp2.resize(equalparts);
    const int binWidth = grayImg_.cols/equalparts;
    for(int i = 0; i < unPx_[0].size(); ++i) {
        const Eigen::Vector2d &upx = unPx_[0][i];
        landmark_.push_back({upx, make_shared<Pose>(), cam_, 1.0});
        // 给地图点赋值描述子
        landmark_.back().descriptor_ = descriptor_[i];
        landmark_.back().UpdateUncertainty();
    }

/*
    // 三角化及校验步骤
    // 遍历每一个kp1，以期生成地图点
    for(int i = 0; i < unPx_[0].size(); ++i) {
        vector<Eigen::Vector2d> debugPositiveDepthKp2;
        vector<double> debugAng;

        double minZ = DBL_MAX;
        double maxZ = -1;
        const Eigen::Vector2d &upx = unPx_[0][i];
        // const Eigen::Matrix<float, kDescriptorPatchSize, 1> d1 = descriptor_[i];
        const int d1 = descriptor_[i];
        const vector<Eigen::Vector2d> kp2 = kf2.FindMatches(upx, Twc_);

        // DrawMatch(edgeImg_[0], kf2.edgeImg_[0], {upx}, kp2, "current point 2 all Epipolar constraint matches", 1, 1);

        for(int j = 0; j < kp2.size(); ++j) {
            const Eigen::Vector2d &p2 = kp2[j];
            const int descId = pointMapId_[{p2.x(), p2.y()}];
            // const Eigen::Matrix<float, kDescriptorPatchSize, 1> d2 = descriptor_[descId];
            const int d2 = descriptor_[descId];
            // const double score = CalculateScore(d1, d2);
            const int score = CalculateDescriptorScore(d1, d2);
            if(score > kMaxDescriptorDist) {
                continue;
            }
            Eigen::Vector3d pc1 = ::Triangulate(upx, p2, T21, *cam_);
            
            // 检验pc1深度值
            if(pc1.z() > kMinDepth && pc1.z() < kMaxDepth) {
                // pc2深度值也要经过校验
                const Eigen::Vector3d pc2 = T21 * pc1;
                if(pc2.z() < kMinDepth || pc2.z() > kMaxDepth) {
                    continue;
                }

                const Eigen::Vector3d po1 = pc1 - Eigen::Vector3d::Zero();
                const Eigen::Vector3d po2 = pc1 - T12.t_wb_;
                // a*b = |a|*|b|*cos(θ)
                const double theta = acos(po1.dot(po2)/po1.norm()/po2.norm() );
                const double ang = abs(theta) * kRad2Deg;
                // 检验视差角，必须保证一定的基线，对于不同方位的地图点而言，基线是不一样的
                cout << j << "score | pc1.z | ang: " << score << " " << pc1.z() << " " << ang << endl;

                if(ang < kMinGoodTriangulateAngle || ang > kMaxGoodTriangulateAngle) {
                    continue;
                }

                // 深度滤波器使用
                if(minZ > pc1.z() ) {
                    minZ = pc1.z();
                }
                if(maxZ < pc1.z() ) {
                    maxZ = pc1.z();
                }
                debugPositiveDepthKp2.push_back(p2.cast<double>());
            }
        }

        // 查看当前p1是否可以三角化出正确点
        if(maxZ < 0 || minZ > kMaxDepth) {
            cerr << "current px1 triangulation error, continue!" << endl;
            continue;
        } else {
            // 这里我们认为首帧是世界帧，所以landmark的anchor帧pose设置为单位矩阵
            landmark_.push_back({upx, make_shared<Pose>(), cam_, 1.0});
            landmark_.back().depthRange_[0] = minZ;
            landmark_.back().depthRange_[1] = maxZ;
            landmark_.back().UpdateUncertainty();

            debugGoodKp1[upx.x() / binWidth].push_back(upx);
            debugGoodKp2[upx.x() / binWidth].push_back(debugPositiveDepthKp2.back());
            // DrawMatch(edgeImg_[0], kf2.edgeImg_[0], {upx}, debugPositiveDepthKp2, "Positive depth Epipolar constraint matches", 1, 1);
            // DrawMatch(edgeImg_[0], kf2.edgeImg_[0], {upx}, {kp2[bestPId2]}, "best match Epipolar constraint matches", 1, 1);
            cout << "\n\n";
        }
    }
*/
    return landmark_.size();
}

void KeyFrame::UpdateDepth(const KeyFrame &kf2) {
    const Pose T21 = kf2.Twc_.Inverse() * Twc_;
    cout << "T12: " << T21.Inverse() << endl;
    for(int i = 0; i < landmark_.size(); ++i) {
        Landmark &pc1 = landmark_[i];
        const vector<Eigen::Vector2d> kp2 = ::FindMatches(pc1, kf2, T21, *cam_);
        
        if(UpdateLandmarkDepth(kp2, T21, *cam_, pc1) ) {
            cout << "[" << pc1.depthRange_[0] << " " << pc1.depthRange_[1] << "] " << pc1.z_ << endl;
            // DrawMatch(edgeImg_[0], kf2.edgeImg_[0], {pc1.uv_}, kp2, "current point 2 all Epipolar constraint matches", 1, 1);
        }   
    }
}
