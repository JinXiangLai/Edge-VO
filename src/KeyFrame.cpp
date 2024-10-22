#include "KeyFrame.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <opencv2/highgui.hpp>
#include <string>
#include <vector>

#include "Config.h"
#include "Eigen/src/Core/Matrix.h"
#include "Pose.h"
#include "Utils.h"

using namespace std;
using namespace cv;

class Landmark;

KeyFrame::KeyFrame(const KeyFrame &f)
    : id_(f.id_)
    , grayImg_(f.grayImg_)
    , edgeImg_(f.edgeImg_)
    , dist_(f.dist_)
    , dx_(f.dx_)
    , dy_(f.dy_)
    , cam_(f.cam_)
    , Twc_(f.Twc_)
    , Tcw_(f.Tcw_)
    , priorTwc_(f.priorTwc_)
    , level_(f.level_)
    , unPx_(f.unPx_)
    , descriptor_(f.descriptor_)
    , pointMapId_(f.pointMapId_)
    , outOfRange_(f.outOfRange_)
    , convergeEdgeNum_(f.convergeEdgeNum_) {
    landmark_.reserve(f.landmark_.size());
    for(Landmark *lk : f.landmark_) {
        landmark_.push_back(new Landmark(*lk));
        // !!!Attention: 指针成员变量需要小心处理，因为如果其指向栈内存，由于栈内存会被系统回收，
        // 因此可能产生意外情况
        landmark_.back()->host_ = this;
    }
}

void KeyFrame::operator =(const KeyFrame &f) {
    new (this) KeyFrame(f);
}


void KeyFrame::CannyEdgeDetect() {
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    cv::Mat blurred;
    // 应用高斯滤波来平滑边缘
    cv::GaussianBlur(grayImg_, blurred, cv::Size(5, 5), 1);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();

    const double lowerThreshold = config->cannyLowerTh; // 下限阈值
    const double upperThreshold = config->cannyupperTh; // 上限阈值，越小提取边缘越多
    int apertureSize = 3;        // 应用Sobel算子的窗口大小
    Canny(blurred, edgeImg_[0], lowerThreshold, upperThreshold, apertureSize);
    chrono::steady_clock::time_point t3 = chrono::steady_clock::now();
    
    if(config->messageLevel == MessageLevel::Debug)
        cout << "Extract canny edge spend " << chrono::duration<double>(t3 - t2).count() << "s" 
                << " & Gaussian Blur spend " << chrono::duration<double>(t2 - t1).count() << endl;

    vector<Point2i> px;
    // 取出边缘像素点
    constexpr int jump = 6;
    for(int x = jump; x < edgeImg_[0].cols-jump; ++x) {
        for(int y = jump; y < edgeImg_[0].rows-jump; ++y) {
            if(edgeImg_[0].at<uchar>(y, x) == 255 && IsFastPoint(grayImg_, {x, y})) {
                px.push_back({x, y});
            }
        }
    }

    //cv::imshow("distor", edgeImg_[0]);
    //cv::waitKey(0);

    chrono::steady_clock::time_point t4 = chrono::steady_clock::now();
    unPx_[0] = cam_->UndistortPoints(px); // 去畸变后的像素平面上的点
    chrono::steady_clock::time_point t5 = chrono::steady_clock::now();
    if(config->messageLevel == MessageLevel::Debug)
        cout << "Undistort " << px.size() << "points spend " 
            << chrono::duration<double>(t5 - t4).count() << "s" << endl;

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

    //cv::imshow("undistor", edgeImg_[0]);
    //cv::waitKey(0);
}

void KeyFrame::GenerateDTandDerivative() {
    dist_[0] = GetDistanceTransform(edgeImg_[0]);
    CaculateDerivative(dist_[0], dx_[0], dy_[0]);
}

size_t KeyFrame::GenerateLandmark(KeyFrame &kf2, vector<vector<Eigen::Vector2d> > &debugGoodKp1, vector<vector<Eigen::Vector2d> >&debugGoodKp2,
    const int equalparts) {
    
    const Pose T21 = kf2.Tcw_ * Twc_;
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
    convergeEdgeNum_ = 0; // 重新统计当前KF的收敛边缘点集

    const Pose Tc1c2 = priorTwc_.Inverse() * kf2.priorTwc_;
    if(Tc1c2.t_wb_.norm() < 0.01) {
        // 位移过小，不能进行更新
        return 0;
    }

    for(int i = 0; i < landmark_.size(); ++i) {
        if(landmark_[i] == nullptr || landmark_[i]->IsOutOfRange()) {
            continue;
        }
        Landmark *lk1 = landmark_[i];
        if(lk1->Converge()) {
            convergeEdgeNum_ += 1;
        }
        // 每个Landmark只能由一个host控制，在转移控制权之前，只能更新其在host系下的depth
        const vector<Eigen::Vector2d> kp2 = lk1->FindMatches(kf2);
        // 更新的是host帧下的深度
        const Pose T21 = kf2.Tcw_ * lk1->host_->Twc_;
        if(UpdateLandmarkDepth(kp2, T21, *cam_, *lk1) ) {
            // cout << "depth range, depth, std: [" << lk1->depthRange_[0] << " " << lk1->depthRange_[1] << "] "
            //  << lk1->z_ << " " << lk1->uncertainty_ << endl;
            // DrawMatch(edgeImg_[0], kf2.edgeImg_[0], {lk1->uv_}, kp2, "current point 2 all Epipolar constraint matches", 1, 1);
            ++lk1->obvTime_; // 对于KF有用，因其会多次更新depth
            if(lk1->Converge() ) {
                matchEdgeNum += 1.0;
            }
        }   
    }
    
    if(config->messageLevel <= MessageLevel::Error)
        cout << setprecision(3) << "matchEdgeNum, convergeEdgeNum_: " << matchEdgeNum 
            << " " << convergeEdgeNum_ << endl;

    if(convergeEdgeNum_ < landmark_.size() * 0.2) {
        // 有效路标点数量过低，需要继续进行深度滤波
        return 0.;
    }
    return double(convergeEdgeNum_) / landmark_.size();
}

// 使用极线约束跟踪每一个边缘点
int KeyFrame::TrackLandmarkByEpilorLine(const KeyFrame &kf1) {
    int trackNum = 0;

    const Pose Tc1c2 = kf1.priorTwc_.Inverse() * priorTwc_;
    if(Tc1c2.t_wb_.norm() < 0.01) {
        // 位移过小，不能进行更新
        return 1;
    }

    const Pose T21 = Tcw_ * kf1.Twc_;

    // 给新的KF2预分配内存
    if(landmark_.empty()) {
        landmark_ = vector<Landmark*>(unPx_[0].size(), nullptr);
    }

    int epipolarSearchNoneNum = 0; // debug参数
    int depthOutofRangeNum = 0;

    const vector<Landmark*> &landmark = kf1.landmark_;
    for(int i = 0; i < landmark.size(); ++i) {
        if(landmark[i] == nullptr || landmark[i]->IsOutOfRange()) {
            // 未收敛的landmark也当作可追踪的，因其在未来可能收敛
            continue;
        }
        Landmark *lp1 = landmark[i];
        const vector<Eigen::Vector2d> kp2 = lp1->FindMatches(*this);
        if(kp2.empty()) {
            ++epipolarSearchNoneNum;
            continue;
        }

        for(const Eigen::Vector2d &p : kp2) {
            const Eigen::Vector3d pc1 = Triangulate(lp1->uv_, p, T21, *cam_);
            if(pc1.z() >= lp1->depthRange_[0] && pc1.z() <= lp1->depthRange_[1]) {
                const int vecId = pointMapId_[p.cast<int>()];
                if(landmark_[vecId] != nullptr) {
                    // TODO: 选一个更好的，或者按照先来后到
                    break;
                }
                const uint64_t d2 = descriptor_[vecId];
                const uint64_t d1 = lp1->descriptor_;
                uint64_t score = CalculateDescriptorScore(d1, d2);
                if(score < config->goodDescriptorDist) {
                    // 增加相互观测
                    landmark_[vecId] = lp1;
                    lp1->target_.insert({this, p.cast<double>()});
                    ++trackNum;

                }
            } else {
                ++depthOutofRangeNum;
            }
        }
    }
    cout << "epipolarSearchNoneNum, depthOutofRangeNum, trackNum: " << epipolarSearchNoneNum
         << " " << depthOutofRangeNum << " " << trackNum << endl;
    return trackNum;
}

int KeyFrame::ReuseLandmark(KeyFrame *kf1) {
    // 给新的KF2预分配内存
    landmark_ = vector<Landmark*>(unPx_[0].size(), nullptr);


    auto Project2Edge = [this](const Eigen::Vector2i px) -> bool {
        // 允许的像素偏差
        const vector<Eigen::Vector2i> xy = {{0, 1}, {0, -1}, {-1, 0}, {1, 0}, 
            {-1, 1}, {1, 1}, {-1, -1}, {1, -1}};
        for(const Eigen::Vector2i &p : xy) {
            Eigen::Vector2i px2 = px + p;
            if(pointMapId_.count(px2) ) {
                return true;
            }
        }
        return false;
    };

    const int debugBin = 5;
    vector<vector<Eigen::Vector2d> > debugProj1(5), debugProj2(5);
    const int debugPart = edgeImg_[0].cols / 5; // 显示分区 

    int reuseLandmarkNum = 0;
    const vector<Landmark*> &landmark = kf1->landmark_;
    for(int i = 0; i < landmark.size(); ++i) {
        if(landmark[i] == nullptr || landmark[i]->IsOutOfRange()) {
            // 未收敛的landmark也当作可追踪的，因其在未来可能收敛
            continue;
        }
        Landmark *pc1 = landmark[i];

        // 更新的是host帧下的depth
        const Eigen::Vector3d pc2 = Tcw_ * pc1->GetPw();
        if(pc2.z() < config->minDepth || pc2.z() > config->maxDepth) {
            continue;
        }

        const Eigen::Vector2i px2 = cam_->Project2PixelPlane(pc2).cast<int>();
        if(pointMapId_.count(px2) || Project2Edge(px2) ) {
            const int vecId = pointMapId_[px2];
            if(landmark_[vecId] != nullptr) {
                // TODO: 选一个更好的，或者按照先来后到
                continue;
            }
            const uint64_t d2 = descriptor_[vecId];
            const uint64_t d1 = pc1->descriptor_;
            uint64_t score = CalculateDescriptorScore(d1, d2);
            if(score < config->goodDescriptorDist) {
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
    cv::destroyAllWindows();
    return reuseLandmarkNum;
}

void KeyFrame::Update(const Eigen::Vector3d &delta_q, const Eigen::Vector3d &delta_t) {
    Twc_.Update(delta_q, delta_t);
    Tcw_ = Twc_.Inverse();
}

void KeyFrame::SetTwc(const Pose &Twc) {
    Twc_ = Twc;
    Tcw_ = Twc.Inverse();
}

double KeyFrame::CullingBadDepth(KeyFrame *kf2) {
    int convergeNum = 0, badNum = 0;
    const Pose T21 = kf2->Twc_.Inverse() * Twc_;
    for(int i = 0; i < landmark_.size(); ++i) {
        Landmark *lk1 = landmark_[i];
        if(lk1==nullptr || lk1->IsOutOfRange() || !lk1->Converge() ) {
            continue;
        }

        ++convergeNum;
        const Eigen::Vector3d pc2 = T21 * lk1->GetPc();
        const Eigen::Vector2i px2 = cam_->Project2PixelPlane(pc2).cast<int>();
        if(!InRange(kf2->grayImg_, px2)) {
            // 投影点不在视野内是正常的
            continue;
        }
        if(kf2->dist_[0].at<float>(px2.y(), px2.x()) > config->maxTrackProjectPixelError) {
            lk1->SetOutOfRange();
            ++badNum;
            continue;
        } 
#ifdef USE_SSD
        // 考虑到图像远近，似乎不能使用这个条件？
        // if( CalculatePatchSSD(grayImg_, kf2->grayImg_, lk1->uv_.cast<int>(), px2) > config->maxSSDdist) {
        //     lk1->SetOutOfRange();
        //     ++badNum;
        //     continue;  
        // }
#endif
    }

    return double(badNum) / convergeNum;
}
