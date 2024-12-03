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

KeyFrame::KeyFrame(const Mat &img, const Pose &Twc, std::shared_ptr<Camera> cam, const int id, const int level)
    : id_(id)
    , grayImg_(img)
    , cam_(cam)
    , Twc_ {Twc}
    , Tcw_(Twc.Inverse())
    , priorTwc_(Twc)
    , level_(level) {
        edgeImg_.resize(level);
        dist_.resize(level);
        dx_.resize(level);
        dy_.resize(level);
        unPx_.resize(level);
}

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
    // vector内的堆内存需要先释放
    // 不能这样子，这是构造函数，默认的内存应该是干净的，
    // 否则你应该调用赋值构造
    for(Landmark *lk : landmark_) {
        if(lk != nullptr) {
            delete lk;
        }
    }
    landmark_.clear();
    landmark_.reserve(f.landmark_.size());
    for(Landmark *lk : f.landmark_) {
        landmark_.push_back(new Landmark(*lk));
        // !!!Attention: 指针成员变量需要小心处理，因为如果其指向栈内存，由于栈内存会被系统回收，
        // 因此可能产生意外情况
        landmark_.back()->host_ = this;
    }
    debugGrayImg_ = f.debugGrayImg_;
}

KeyFrame::~KeyFrame() {
    // 由于Landmar与KeyFrame相互引用，所以之前将析构函数放在头文件导致landmark_内存无法释放？？
    for(Landmark *lk : landmark_) {
        if(lk!=nullptr) {
            delete lk;
            lk = nullptr;
        }
    }
    // ReleaseMat(); // 不需要手动释放
    std::cout << this << " Releasw KF id: " << id_ << std::endl;
}

void KeyFrame::operator =(const KeyFrame &f) {
#if 1
    id_ = f.id_;
    grayImg_ = f.grayImg_;
    edgeImg_ = f.edgeImg_;
    dist_ = f.dist_;
    dx_ = f.dx_;
    dy_ = f.dy_;
    cam_ = f.cam_;
    Twc_ = f.Twc_;
    Tcw_ = f.Tcw_;
    priorTwc_ = f.priorTwc_;
    level_ = f.level_;
    unPx_ = f.unPx_;
    descriptor_ = f.descriptor_;
    pointMapId_ = f.pointMapId_;
    outOfRange_ = f.outOfRange_;
    convergeEdgeNum_ = f.convergeEdgeNum_;
    for(Landmark *lk : landmark_) {
        if(lk != nullptr) {
            delete lk;
        }
    }
    landmark_.clear();
    landmark_.reserve(f.landmark_.size());
    for(Landmark *lk : f.landmark_) {
        landmark_.push_back(new Landmark(*lk));
        // !!!Attention: 指针成员变量需要小心处理，因为如果其指向栈内存，由于栈内存会被系统回收，
        // 因此可能产生意外情况
        landmark_.back()->host_ = this;
    }
#else
    ReleaseMat();
    // 这样会导致cv::Mat等堆内存无法释放
    new (this) KeyFrame(f);
#endif
}


void KeyFrame::CannyEdgeDetect() {
#define Undistort

#ifdef Undistort
    Mat D;
    if(config->model == "fisheye") {
        D = (cv::Mat_<float>(4, 1) << cam_->k1_, cam_->k2_, cam_->k3_, cam_->k4_);
    } else if (config->model == "pinhole") {
        D = (cv::Mat_<float>(5, 1) << cam_->k1_, cam_->k2_, cam_->k3_, cam_->k4_, cam_->k5_);
    }
    Mat R = cv::Mat::eye(3, 3, CV_32F);
    Mat K = (cv::Mat_<float>(3, 3) << cam_->fx_, 0, cam_->cx_, 0, cam_->fy_ , cam_->cy_, 0, 0, 1);
    Mat map1, map2;
    cv::initUndistortRectifyMap(K, D, cv::Mat(), K,
		cv::Size(grayImg_.cols, grayImg_.rows), CV_8UC1, map1, map2);
	cv::remap(grayImg_, grayImg_, map1, map2, cv::INTER_LINEAR);
#endif

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
#if 1
    cv::Mat blurred = grayImg_.clone();
    // 应用高斯滤波来平滑边缘
    cv::GaussianBlur(grayImg_, blurred, cv::Size(5, 5), 1);
#else
    Mat blurred = grayImg_.clone();
#endif
    debugGrayImg_ = grayImg_.clone();
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();

    const double lowerThreshold = config->cannyLowerTh; // 下限阈值
    const double upperThreshold = config->cannyupperTh; // 上限阈值，越小提取边缘越多
    int apertureSize = 3;        // 应用Sobel算子的窗口大小

    for(int lvl = 0; lvl < level_; ++lvl) {
        Mat blur_i;
        double ratio = 1./pow(2, lvl);
        if(lvl > 0)
            cv::resize(blurred, blur_i, cv::Size(ratio * blurred.cols, ratio * blurred.rows));
        else
            blur_i = blurred;

        Canny(blur_i, edgeImg_[lvl], lowerThreshold, upperThreshold, apertureSize);
        // 膨胀及腐蚀操作，看起来收益不大
        // cv::Mat er = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        //cv::Mat di = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(8, 8));
        // cv::erode(blur_i, blur_i, er);
        //cv::dilate(blur_i, blur_i, di);
        
        chrono::steady_clock::time_point t3 = chrono::steady_clock::now();
        
        if(config->messageLevel == MessageLevel::Debug)
            cout << "Extract canny edge spend " << chrono::duration<double>(t3 - t2).count() << "s" 
                    << " & Gaussian Blur spend " << chrono::duration<double>(t2 - t1).count() << endl;

        vector<Point2i> px;
        // 取出边缘像素点
        // 这里跳过了6个图像边缘的像素
        constexpr int jump = 2;
        for(int x = jump; x < edgeImg_[lvl].cols-jump; ++x) {
            for(int y = jump; y < edgeImg_[lvl].rows-jump; ++y) {
                if(edgeImg_[lvl].at<uchar>(y, x) != 0 && ( lvl!=0 || IsFastPoint(grayImg_, {x, y}) )) {
                    px.push_back({x, y});
                }
            }
        }

        // cv::imshow("distor", edgeImg_[i]);
        // cv::waitKey(0);

        chrono::steady_clock::time_point t4 = chrono::steady_clock::now();
#ifdef Undistort
        unPx_[lvl].clear();
        for(const Point2i &p : px) {
            unPx_[lvl].push_back({p.x, p.y});
        }
#else
        unPx_[lvl] = cam_->UndistortPoints(px, lvl); // 去畸变后的像素平面上的点
#endif
        chrono::steady_clock::time_point t5 = chrono::steady_clock::now();
        if(config->messageLevel == MessageLevel::Debug)
            cout << "Undistort " << px.size() << "points spend " 
                << chrono::duration<double>(t5 - t4).count() << "s" << endl;

        edgeImg_[lvl] = Mat::ones(edgeImg_[lvl].rows, edgeImg_[lvl].cols, CV_8UC1) * 255;
        vector<Eigen::Vector2d>::iterator it = unPx_[lvl].begin();
        int id = -1;
        while(it != unPx_[lvl].end()) {
            ++id;
            const Eigen::Vector2d &p = *it;
            // 边缘点置为黑色
            if(InRange(edgeImg_[lvl], {p.x(), p.y()}) ) {
                edgeImg_[lvl].at<uchar>(int(p.y()), int(p.x()) ) = 0;
                if(lvl == 0) {
                    debugGrayImg_.at<uchar>(int(p.y()), int(p.x()) ) = 255;
                }
                // 使用未去畸变像素邻域
                const int x = px[id].x, y = px[id].y;
                
                // 计算描述子
                // const Mat &m = grayImg_;
                // Eigen::Matrix<float, kDescriptorPatchSize, 1> d;
                // d << m.at<uchar>(y-1, x-1), m.at<uchar>(y-1, x), m.at<uchar>(y-1, x+1),
                //      m.at<uchar>(y, x-1), m.at<uchar>(y, x), m.at<uchar>(y, x+1), 
                //      m.at<uchar>(y+1, x-1), m.at<uchar>(y+1, x), m.at<uchar>(y+1, x+1);
                // descriptor_.emplace_back(d);

                // 只保留第0层金字塔
                if(lvl == 0) {
                    descriptor_.push_back(::CalculateDescriptor(grayImg_, {x, y}) );
                    pointMapId_.insert({ {p.x(), p.y()}, descriptor_.size()-1});

                }
                ++it;
            } else {
                it = unPx_[lvl].erase(it);
            }
        }

        //cv::imshow("undistor", edgeImg_[i]);
        //cv::waitKey(0);

    }
/*
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
            debugGrayImg_.at<uchar>(int(p.y()), int(p.x()) ) = 255;
            // 使用未去畸变像素邻域
            const int x = px[id].x, y = px[id].y;
            
            // 计算描述子
            // const Mat &m = grayImg_;
            // Eigen::Matrix<float, kDescriptorPatchSize, 1> d;
            // d << m.at<uchar>(y-1, x-1), m.at<uchar>(y-1, x), m.at<uchar>(y-1, x+1),
            //      m.at<uchar>(y, x-1), m.at<uchar>(y, x), m.at<uchar>(y, x+1), 
            //      m.at<uchar>(y+1, x-1), m.at<uchar>(y+1, x), m.at<uchar>(y+1, x+1);
            // descriptor_.emplace_back(d);

            descriptor_.push_back(::CalculateDescriptor(grayImg_, {x, y}) );
            pointMapId_.insert({ {p.x(), p.y()}, descriptor_.size()-1});
            ++it;
        } else {
            it = unPx_[0].erase(it);
        }
    }

    //cv::imshow("undistor", debugGrayImg_);
    //cv::waitKey(0);
*/
}

void KeyFrame::GenerateDTandDerivative() {
    for(int lvl = 0; lvl < level_; ++lvl) {
        dist_[lvl] = GetDistanceTransform(edgeImg_[lvl]);
        CaculateDerivative(dist_[lvl], dx_[lvl], dy_[lvl]);
    }
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
        const uint64_t descriptor = 0;
        landmark_[i] = new Landmark(unPx_[0][i], this, cam_, descriptor, 1.0);
        // host帧也要增加与landmark的相互观测
        landmark_[i]->target_.insert({this, unPx_[0][i]}); 
    }

    return landmark_.size();
}

double KeyFrame::UpdateDepth(const KeyFrame &kf2) {
    double matchEdgeNum = 0;
    convergeEdgeNum_ = 0; // 重新统计当前KF的收敛边缘点集
    int findMatchNum = 0;

    const Pose Tc1c2 = priorTwc_.Inverse() * kf2.priorTwc_;
    if(Tc1c2.t_wb_.norm() < 0.001) {
        // 位移过小，不能进行更新
        return 0;
    }

    // vector<Pose> vTwc{ Pose(), Tc1c2}; 
    // vector<Mat> imgs{ debugGrayImg_, kf2.debugGrayImg_ };
    // ShowCameraCone(vTwc, imgs, *cam_);

    for(int i = 0; i < landmark_.size(); ++i) {
        if(landmark_[i] == nullptr || landmark_[i]->IsOutOfRange()) {
            continue;
        }
        Landmark *lk1 = landmark_[i];
        if(lk1->Converge()) {
            convergeEdgeNum_ += 1;
        }
        // 每个Landmark只能由一个host控制，在转移控制权之前，只能更新其在host系下的depth
        // const vector<Eigen::Vector2d> kp2 = lk1->FindMatches(kf2);
        double bestDepth, std;
        Eigen::Vector2d bestPx2;
        const double error = FindMatchesWithEpipolarConstraintOnImagePlane(&kf2, lk1, bestDepth, std, bestPx2);
        if(error == EpipolarMatchType::outOFboundaryORabnormalDepth) {
            continue;

        } else if(error == EpipolarMatchType::repeatTextureORbadDepth) {
            ++lk1->failObvTime_;
            lk1->depthCov_ *= 1.01;
            lk1->UpdateUncertainty(false);
            continue;

        } else if(error == EpipolarMatchType::occulsionORnoBestMatch) {        
            lk1->depthCov_ *= 1.01;
            lk1->UpdateUncertainty(false);
            continue;

        } else {
            const double diff = lk1->z_ - bestDepth;
            if(diff*diff > std*std + lk1->depthCov_) {
                lk1->depthCov_ *= 1.1;
                lk1->UpdateUncertainty(false);
                continue;
            }

            if(!CheckDepthQuality(*lk1, Tc1c2, bestPx2, bestDepth)) {
                continue;
            }

            double u2 = bestDepth, cov2 = std * std; // 考虑基线的影响
            double u1 = lk1->z_, cov1 = lk1->depthCov_ * 1.01; 
            if(lk1->obvTime_ == 0) {
                // 首次初始化
                u1 = u2;
                cov1 = cov2 * 9;
                // cov2 = cov1 * 0.25; // 不完全信赖第一次的三角化
            } 

            lk1->z_ = (u2*cov1 + u1*cov2) / (cov1 + cov2);
            lk1->depthCov_ = (cov1 * cov2)/(cov1 + cov2);
            //cout << "u1, u2, cov1, cov2, z: " << u1 << " " << u2 << " " << cov1 << " " << cov2 
            //     << " " << landmark.z_ << endl;
            lk1->UpdateUncertainty(true);
            ++findMatchNum;
        }
    }
    
    if(config->messageLevel <= MessageLevel::Error)
        cout << setprecision(3) << "matchEdgeNum, convergeEdgeNum_: " << matchEdgeNum 
            << " " << convergeEdgeNum_ << endl;
    cout << "successful findMatchNum: " << findMatchNum << endl;
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

        bool isBad = false;
        const double residual = abs(grayImg_.at<uchar>(lk1->uv_.y(), lk1->uv_.x()) - kf2->grayImg_.at<uchar>(px2.y(), px2.x()) );
        if(residual > config->maxDescriptorDist) {
            // lk1->depthCov_ *= 1.1;
            // isBad = true;
        }

        if(kf2->dist_[0].at<float>(px2.y(), px2.x()) > config->maxTrackProjectPixelError) {
            // lk1->SetOutOfRange();
            // 投影点误差大的就给它重置
            // lk1->depthCov_ = pow(config->maxDepth, 2);
            lk1->depthCov_ *= 2.0;
            isBad = true;
        } 

        if(isBad) {
            ++badNum;
            lk1->UpdateUncertainty(false);
        }


#ifdef USE_SSD
        // 考虑到图像远近，似乎不能使用这个条件？
        // if( CalculatePatchSSD(this, kf2, lk1->uv_.cast<int>(), px2) > config->maxSSDdist) {
        //     lk1->SetOutOfRange();
        //     ++badNum;
        //     continue;  
        // }
#endif
    }

    return double(badNum) / convergeNum;
}


double KeyFrame::FindMatchesWithEpipolarConstraintOnImagePlane(const KeyFrame* kf2, Landmark* lk1, double &bestDepth, double &std, Eigen::Vector2d &bestPx2) {
    if(lk1 == nullptr || lk1->IsOutOfRange()) {
        return EpipolarMatchType::nanValueNOstereoVisionIssue;
    }

    const Pose T21 = kf2->Twc_.Inverse() * Twc_;
    const Pose T12 = T21.Inverse();
    const double fx = cam_->fx_, fy = cam_->fy_, cx = cam_->cx_, cy = cam_->cy_;

    // 求KF1像素平面上对应极线，直接把O2当作O1相机系下的一点，那么极点位置不就轻松求出来了
    // 已知 t12, 那么KF2光心与KF1归一化平面的交点可求，但是当z[2] = 0时呢？
    const Eigen::Vector3d &t12 = T12.t_wb_;
    Eigen::Vector2d p1 = lk1->uv_;
    // ep1 = (x, y)*t12.z - 极点e1*t12.z
    // 注意：这个极线是像素平面上放大 t12.z 倍后的方向向量
    Eigen::Vector2d ep1{ -fx*t12[0] + t12[2]*(p1.x()-cx), -fy*t12[1] + t12[2]*(p1.y()-cy) };
    // 要保证双目图像上极线方向相对于图像是从左到右还是从右到左保持一致
    // ep1 *= -1;
    if(ep1.norm() < 1.0) {
        return EpipolarMatchType::nanValueNOstereoVisionIssue;
    }
    ep1.normalize();

    const Eigen::Vector3d priorPc2 = T21 * lk1->GetPc();
    const double depthScale = priorPc2.z() / lk1->z_;
    if(!(depthScale > 0.7f && depthScale < 1.4f))
	{
        return EpipolarMatchType::outOFboundaryORabnormalDepth;
	}
    // 根据近大远小的规则调整窗口范围
    ep1 *= depthScale; 
    const int desLen = config->descriptorPatchLen;
    const int midLen = desLen / 2;
    // 检验一下描述子是否在范围内
    Eigen::Vector2d p1Start = p1 - midLen*ep1, p1End = p1 + midLen*ep1;       
    if(!InRange(grayImg_, p1Start.cast<int>()) || !InRange(grayImg_, p1End.cast<int>())) {
        return EpipolarMatchType::outOFboundaryORabnormalDepth;
    }
    vector<Eigen::Vector2d> debugPx1{p1};


    double maxZ1 = lk1->z_+3*lk1->uncertainty_;
    double minZ1 = max(0.1, lk1->z_-3*lk1->uncertainty_); 
    const Eigen::Vector3d farPc1 = cam_->InverseProject(lk1->uv_.cast<int>(), maxZ1);
    const Eigen::Vector3d nearPc1 = cam_->InverseProject(lk1->uv_.cast<int>(), minZ1);
    const Eigen::Vector3d farPc2 = T21 * farPc1;
    const Eigen::Vector3d nearPc2 = T21 * nearPc1;
    if(farPc2.z() < nearPc2.z() ) {
        return EpipolarMatchType::outOFboundaryORabnormalDepth;
    }
    Eigen::Vector2d farPx2 = cam_->Project2PixelPlane(farPc2),
                    nearPx2 = cam_->Project2PixelPlane(nearPc2);
    // 从 far->near 的方向向量，ep1极线向量的方向相对图像需要与此保持一致
    Eigen::Vector2d ep2 = nearPx2 - farPx2; 
    const double ep2Len = ep2.norm();
    ep2 = ep2.normalized();


    // OK，接下来在对极线上等距取5个点，据此来计算SSD
    ep1 *= config->minSearchStep;
    vector<double> v1 = CalculateDescriptor(grayImg_, p1, ep1, desLen);
    double s1 = 0;
    double avg1 = 0;
    for(int i = 0; i < desLen; ++i) {
        s1 += v1[i];
    }
    avg1 = s1/desLen;


    const Mat &edgeImg2 = kf2->edgeImg_[0];

    // OK， 我们需要限制一下KF2上的极线范围，这是最重要的先验，极线搜索距离越短，受相对旋转的影响越小
    const double maxEpipolarLen = config->maxEpipolarSearchLine, minEpipolarLen = config->minEpipolarSearchLine;
    const double cutLen = ep2Len - maxEpipolarLen;
    const double expandLen = minEpipolarLen - ep2Len;
    if(cutLen > 0) {
        // 只能修改最近点
        // nearPx2 -= cutLen * ep2;
        // nearPx2 = farPx2 - cutLen * ep2;
        nearPx2 = farPx2 + maxEpipolarLen*ep2;
    } else if(expandLen > 0) {
        const double halfLen = 0.5 * expandLen;
        nearPx2 += halfLen * ep2;
        farPx2 -= halfLen * ep2;
    }
    // 最远点若不在投影范围内，那么意味着视野范围受限？直接返回
    if(!InRange(edgeImg2, farPx2.cast<int>()) ) {
        return EpipolarMatchType::outOFboundaryORabnormalDepth;
    }
    if(!InRange(edgeImg2, nearPx2.cast<int>()) ) {
        // 将最近点移动到图像内
        if(!MoveNearPx2IntoBoundary(nearPx2, ep2, farPx2) ) {
            return EpipolarMatchType::outOFboundaryORabnormalDepth;
        }
    }

    double bestScore = 1e9, secondBestScore = 1e9;
    Eigen::Vector2d bestP2{1000, 1000}, secondBestP2{1000, 1000};

    Eigen::Vector2d p2 = farPx2;
    ep2 *= config->minSearchStep;
    //cout << "farPx2, ep2: " << p2.transpose() << " | " << ep2.transpose() << endl;
    vector<Eigen::Vector2d> debugPx2{p2};

    Eigen::Vector2d p2End = p2 - midLen*ep2, p2Start = p2 + midLen*ep2;
    // 保证端点在图像范围内
    if(!InRange(edgeImg2, p2End.cast<int>()) ) {
        p2End = p2;
        p2 = p2Start;
        p2Start = p2 + midLen*ep2;

    }
    const Mat &img2 = kf2->grayImg_;
    vector<double> v2 = CalculateDescriptor(img2, p2, ep2, desLen);

    double s2 = 0, avg2 = 0; // 可以使用滑窗计算
    for(int i = 0; i < desLen; ++i) {
        s2 += v2[i];
    }
    avg2 = s2/desLen;

    if(ep2.norm() < 1) {
        return EpipolarMatchType::outOFboundaryORabnormalDepth;
    }

    // 通过判断incx，incy的正负，来判断循环终止条件
    // while ( (ep2[0] > 0) == (p2[0] < nearPx2[0]) && (ep2[1] > 0) == (p2[1] < nearPx2[1]) ) {
    while (1) {
        //cout << "p2m2, p2p2: " << p2m2.transpose() << " | " << p2p2.transpose() << endl;
        if (1) {

            // 已经保证端点在边界范围内，这里无需再判断
            const double score = CalculateSSD(&v1[0], &v2[0], avg1, avg2, desLen);

            if(score < bestScore) {
                secondBestScore = bestScore;
                secondBestP2 = bestP2;
                bestScore = score;
                bestP2 = p2;
            } else if(score < secondBestScore) {
                secondBestScore = score;
                secondBestP2 = p2;
            }
        }

        // 根据滑窗的思路，我们可以只移动一个即可
        p2End += ep2; // 判断边界
        p2 += ep2; // 移动关键点
        p2Start += ep2; // 判断边界及移动滑窗
        s2 -= v2[0];
        for(int i = 1; i < v2.size(); ++i) {
            v2[i-1] = v2[i];
        }
        const bool inSearchRange = (ep2[0] > 0) == (p2Start[0] < nearPx2[0]) && (ep2[1] > 0) == (p2Start[1] < nearPx2[1]);
        if(!inSearchRange) {
            break;
        }
        const double v2Back = BilinearInterpolate<uchar>(kf2->grayImg_, p2Start);
        v2.back() = v2Back;
        s2 += v2.back();
        avg2 = s2 / desLen;

        // for debug only
        debugPx2.push_back(p2);
        p1 += ep1;
        debugPx1.push_back(p1);
    }

    vector<Eigen::Vector2d> debugGoodKp2;
    if(InRange(edgeImg2, bestP2.cast<int>())) {
        //cout << "best, second score mean: " << bestScore/desLen << " " << secondBestScore/desLen << endl;
        debugGoodKp2.push_back(bestP2); // yellow
    }
    if(InRange(edgeImg2, secondBestP2.cast<int>())) {
        debugGoodKp2.push_back(secondBestP2);
    }

    // cout << "Each best, second score mean: " << bestScore/desLen << " " << secondBestScore/desLen << endl;

    if(config->drawAllEpipolarMatch && lk1->uv_.x() > config->drawEpipolarMatchStartCol) {
        cout << "best & second score: " << bestScore << " " << secondBestScore << endl;
        cout << "(bestP2 - secondBestP2).norm(): " << (bestP2 - secondBestP2).norm() << endl;
        DrawMatch(debugGrayImg_, kf2->debugGrayImg_, debugPx1, debugPx2, debugGoodKp2, 
            "each point 2 all Epipolar constraint matches", 1, 1000000);
    }

    const bool smallScore = bestScore < config->maxDescriptorDist * desLen;
    if(!smallScore) {
        return EpipolarMatchType::occulsionORnoBestMatch;
    }

    const bool goodScore = bestScore < config->best2SecondRatio * secondBestScore;
    if(!goodScore) {
        return EpipolarMatchType::repeatTextureORbadDepth;
    }

    double badDist = false;
    if(secondBestScore < 1e8) {
        badDist = (bestP2 - secondBestP2).norm() > config->best2SecondDist;
    }
    if(badDist) {
        return EpipolarMatchType::repeatTextureORbadDepth;
    }

    const double maxMeasureDepth = fx * fy * T12.t_wb_.norm()/sqrt(fx*fx+fy*fy); // 1pixel
    const double minMeasureDepth = maxMeasureDepth / config->maxEpipolarSearchLine; 

    // if(pointMapId_.at(lk1->uv_.cast<int>()) == 0)
    static int mea = 0;
    if(mea++%10000==0)
        cout << "maxMeasureDepth, minMeasureDepth: [" << maxMeasureDepth << " " << minMeasureDepth << "]" << endl;


    if(goodScore && !badDist) {
        const int time = 2;
        const double dm1 = TriangulateDepth(lk1->uv_, bestP2-time*ep2, T21, *cam_);
        const double d1 = TriangulateDepth(lk1->uv_, bestP2, T21, *cam_);
        const double dp1 = TriangulateDepth(lk1->uv_, bestP2+time*ep2, T21, *cam_);
        if(d1 < 0 || d1 > maxMeasureDepth || d1 < minMeasureDepth) {
            return EpipolarMatchType::repeatTextureORbadDepth;
        }
        // 对于深度大于5米的，发现深度差异会非常大
        // 对于深度小于0.5米的，结果是 d: [0.829857 0.203911 0.78017] 深度比例也差很大
        const double ratio1 = dm1 > d1? (dm1-d1)/dm1 : (d1-dm1)/d1;
        const double ratio2 = dp1 > d1? (dp1-d1)/dp1 : (d1-dp1)/d1;
        const double uncertainty1 = dm1 > d1? (dm1-d1) : (d1-dm1);
        const double uncertainty2 = dp1 > d1? (dp1-d1) : (d1-dp1);
        const double maxUncertainty = 1e9; // 0.5
        if( ratio1 > 0.6 || ratio2 > 0.6 || uncertainty1 > maxUncertainty || uncertainty2 > maxUncertainty) {
            return EpipolarMatchType::outOFboundaryORabnormalDepth;
        }

        // if(config->drawGoddEpipolarMatch && (lk1->uv_.x() > config->drawEpipolarMatchStartCol || d1 < 0.5)) {
        if(config->drawGoddEpipolarMatch && (d1 < 0.5)) { // KF2上投影得到的极线距离非常短
        // if(config->drawGoddEpipolarMatch && (d1 > 5.0)) {
            cout << " d: [" << dm1 << " " << d1 << " " << dp1 << "]" << endl;
            cout << "parallax: " << (lk1->uv_-bestP2).norm() << endl;
            cout << "bestScore, secondBestScore/desLen: " << bestScore/desLen << " " << secondBestScore/desLen << endl;
            cout << "(bestP2 - secondBestP2).norm(): " << (bestP2 - secondBestP2).norm() << endl;
            DrawMatch(debugGrayImg_, kf2->debugGrayImg_, debugPx1, debugPx2, debugGoodKp2, 
                "current point 2 all Epipolar constraint matches", 1, 1000000);
        }
        
        bestDepth = d1;
        std = max(uncertainty1, uncertainty2) * 2; // 考虑像素测量误差
        bestPx2 = bestP2;

        cout << " d: [" << dm1 << " " << d1 << " " << dp1 << "], uncertainty:[ " << abs(dp1-d1) << " " << abs(dm1-d1) << "]" << endl;
        return bestScore;
    }
    return EpipolarMatchType::outOFboundaryORabnormalDepth;
}

vector<double> KeyFrame::CalculateDescriptor(const cv::Mat &grayImg, const Eigen::Vector2d &px, const Eigen::Vector2d &epNorm, const int len) {
    vector<double> des(len, 0.);
    if(len%2==0) {
        cerr << "descriptor length must be odd number" << endl;
        exit(-1);
    }
    const int mid = len / 2;
    // 这里我们使用双线性插值来获取光度，这样就不用担心四舍五入的问题了
    des[mid] = BilinearInterpolate<uchar>(grayImg, px);
    int incRatio = 1;
    const int maxId = len - 1;
    for(int i = mid-1; i >= 0; --i) {
        des[i] = BilinearInterpolate<uchar>(grayImg, px - incRatio*epNorm);
        des[maxId - i] = BilinearInterpolate<uchar>(grayImg, px + incRatio*epNorm);
        ++incRatio;
    }
    return des;
}


double KeyFrame::CalculateSSD(double *v1, double *v2, double avg1, double avg2, const int desLen) {
    double sum = 0;
    double avg = avg1 - avg2;
    // avg = 0;
    for(int i = 0; i < desLen; ++i) {
        sum += abs(v1[i] - avg - v2[i]);
    }
    return sum;
}

bool KeyFrame::MoveNearPx2IntoBoundary(Eigen::Vector2d &pClose, const Eigen::Vector2d &ep2, const Eigen::Vector2d &pFar) {
#define SAMPLE_POINT_TO_BORDER 7
    const int width = edgeImg_[0].cols,
        height = edgeImg_[0].rows;
    const double incx = ep2[0], incy = ep2[1];
    if(
			pClose[0] <= SAMPLE_POINT_TO_BORDER ||
			pClose[0] >= width-SAMPLE_POINT_TO_BORDER ||
			pClose[1] <= SAMPLE_POINT_TO_BORDER ||
			pClose[1] >= height-SAMPLE_POINT_TO_BORDER)
	{
		if(pClose[0] <= SAMPLE_POINT_TO_BORDER)
		{
			float toAdd = (SAMPLE_POINT_TO_BORDER - pClose[0]) / incx;
			pClose[0] += toAdd * incx;
			pClose[1] += toAdd * incy;
		}
		else if(pClose[0] >= width-SAMPLE_POINT_TO_BORDER)
		{
			float toAdd = (width-SAMPLE_POINT_TO_BORDER - pClose[0]) / incx;
			pClose[0] += toAdd * incx;
			pClose[1] += toAdd * incy;
		}

		if(pClose[1] <= SAMPLE_POINT_TO_BORDER)
		{
			float toAdd = (SAMPLE_POINT_TO_BORDER - pClose[1]) / incy;
			pClose[0] += toAdd * incx;
			pClose[1] += toAdd * incy;
		}
		else if(pClose[1] >= height-SAMPLE_POINT_TO_BORDER)
		{
			float toAdd = (height-SAMPLE_POINT_TO_BORDER - pClose[1]) / incy;
			pClose[0] += toAdd * incx;
			pClose[1] += toAdd * incy;
		}

		// get new epl length
        // OK, 根据极线端点我们就能计算极线长度了
		float fincx = pClose[0] - pFar[0];
		float fincy = pClose[1] - pFar[1];
		float newEplLength = sqrt(fincx*fincx+fincy*fincy);

		// test again
        // 如果curF上对应的极线太短，意味着旋转太大？导致无法找到正确匹配？
        // oob: out of border, 超出范围？
		if(
				pClose[0] <= SAMPLE_POINT_TO_BORDER ||
				pClose[0] >= width-SAMPLE_POINT_TO_BORDER ||
				pClose[1] <= SAMPLE_POINT_TO_BORDER ||
				pClose[1] >= height-SAMPLE_POINT_TO_BORDER ||
				newEplLength < 8.0f
				)
		{
			return false;
		}
	}
    return true;
}


void KeyFrame::ReleaseMat() {
    grayImg_.release();
    for(int i = 0; i < edgeImg_.size(); ++i) {
        edgeImg_[i].release();
        dist_[i].release();
        dx_[i].release();
        dy_[i].release();
    }
}

void KeyFrame::FuseDepth() {
    const int fusePatchLen = 5;
    const int midLen = 5/2;
    static vector<vector<double> > weights(fusePatchLen, vector<double>(fusePatchLen, 0));
    static bool first = true;
    if(first) {
        for(int x = -midLen; x <= midLen; ++x)
            for(int y = -midLen; y <= midLen; ++y) {
                // 权重与距离成反比
                // 避免除以0
                weights[y+midLen][x+midLen] = 1./(x*x + y*y + 1.0);
            }
        first = false;
    }

    KeyFrame temp = *this; // I'm too lazy
    vector<Landmark*> &readLk = temp.landmark_;
    unordered_map<Eigen::Vector2i, int, TupleHash> &readPointMapId = temp.pointMapId_;

    for(int i = 0; i < landmark_.size(); ++i) {
        double maxDepth = -1, minDepth = 1e9;
        Landmark *lk1 = landmark_[i];
        const Eigen::Vector2i px1 = lk1->uv_.cast<int>();
        const double d1 = lk1->z_;

        double sumDepth = 0, sumWeight = 0;
        int sumCount = 0;
        for(int x = -midLen; x <= midLen; ++x) {
            for(int y = -midLen; y <= midLen; ++y) {
                const double w = weights[y+midLen][x+midLen];
                const Eigen::Vector2i px2 = px1 + Eigen::Vector2i(x, y);
                if(!readPointMapId.count(px2) ) {
                    continue;
                }
                Landmark *lk2 = readLk[readPointMapId.at(px2)];
                if(lk2!=nullptr && !lk2->IsOutOfRange() && lk2->Converge()) {
                    const double d2 = lk2->z_;
                    if(abs(d2 - d1) > lk1->uncertainty_ * 2.0) {
                        continue;
                    }
                    if(d2 > maxDepth) {
                        maxDepth = d2;
                    }
                    if(d2 < minDepth) {
                        minDepth = d2;
                    }

                    const double mixWeight = 1/lk2->depthCov_ + w;
                    sumDepth += mixWeight * d2;
                    sumWeight += mixWeight;
                    ++sumCount;
                }
            }     
        }

        if(maxDepth > 0 && minDepth < maxDepth && sumCount > 3) {
            lk1->z_ = sumDepth / sumWeight;
            lk1->depthCov_ = pow(maxDepth - minDepth, 2) ;
            lk1->UpdateUncertainty(true);
        }
    } 
}

