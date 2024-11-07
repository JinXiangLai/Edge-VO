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
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
#if 0
    cv::Mat blurred;
    // 应用高斯滤波来平滑边缘
    cv::GaussianBlur(grayImg_, blurred, cv::Size(5, 5), 1);
#else
    Mat blurred = grayImg_;
#endif
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
        if(landmark_[i] == nullptr || landmark_[i]->IsOutOfRange() || i%20 != 0) {
            continue;
        }
        Landmark *lk1 = landmark_[i];
        if(lk1->Converge()) {
            convergeEdgeNum_ += 1;
        }
        // 每个Landmark只能由一个host控制，在转移控制权之前，只能更新其在host系下的depth
        //const vector<Eigen::Vector2d> kp2 = lk1->FindMatches(kf2);
        const vector<Eigen::Vector2d> kp2 = FindMatchesWithEpipolarConstraintOnImagePlane(&kf2, lk1);
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
        // if( CalculatePatchSSD(this, kf2, lk1->uv_.cast<int>(), px2) > config->maxSSDdist) {
        //     lk1->SetOutOfRange();
        //     ++badNum;
        //     continue;  
        // }
#endif
    }

    return double(badNum) / convergeNum;
}


vector<Eigen::Vector2d> KeyFrame::FindMatchesWithEpipolarConstraintOnImagePlane(const KeyFrame* kf2, Landmark* lk1) {
    const Pose T21 = kf2->Twc_.Inverse() * Twc_;
    const Pose T12 = T21.Inverse();
    const Mat &edgeImg2 = kf2->edgeImg_[0];

    // for(int i = 0; i < landmark_.size(); ++i) {
    //     Landmark *lk1 = landmark_[i];
        if(lk1 == nullptr || lk1->IsOutOfRange()) {
            // continue;
            return {};
        }

        double maxZ1 = lk1->z_+3*lk1->uncertainty_;
        double minZ1 = max(0.1, lk1->z_-3*lk1->uncertainty_); 
        const Eigen::Vector3d farPc1 = cam_->InverseProject(lk1->uv_.cast<int>(), maxZ1);
        const Eigen::Vector3d nearPc1 = cam_->InverseProject(lk1->uv_.cast<int>(), minZ1);
        const Eigen::Vector3d farPc2 = T21 * farPc1;
        if(farPc2.z() < 0.1) {
            return {};
        }
        const Eigen::Vector3d nearPc2 = T21 * nearPc1;
        if(nearPc2.z() < 0.1) {
            return {};
        }
        Eigen::Vector2d farPx2 = cam_->Project2PixelPlane(farPc2),
                              nearPx2 = cam_->Project2PixelPlane(nearPc2);
        // 从 far->near 的方向向量
        Eigen::Vector2d ep2 = nearPx2 - farPx2; // 我们从最远到最近深度进行遍历
        // TODO： 保证一定的极线长度
        // 这里很简单我们就获得了KF2上的极线端点和极线单位向量{1px}
        ep2 /= ep2.norm();

        // OK，接下来求KF1像素平面上对应极线
        // 已知 t12, 那么KF2光心与KF1归一化平面的交点可求，但是当z[2] = 0时呢？
        const Eigen::Vector3d P12 = T12.t_wb_;
        Eigen::Vector2d ep1;
        Eigen::Vector2d px1; // debug极点显示
        if(P12[2] != 0) {
            const Eigen::Vector2d xyNorm = (P12/P12[2]).head(2);
            px1[0] = cam_->fx_ * xyNorm[0] + cam_->cx_;
            px1[1] = cam_->fy_ * xyNorm[1] + cam_->cy_;
            ep1 = px1 - lk1->uv_;
            cout << "px1 | l1: " << px1.transpose() << " | " << l1 << endl;
            cout<< lk1 << "  " << "ep1-1: " << ep1.transpose() << endl;

        } else {
            // 两光心的连线O1O2在一条直线上，所以， KF1上的极线在哪里呢？
            // 答：将KF1画称右手OXYZ世界系，再画其上的z=1平面，由于OXY平面平行于z=1平面，
            // 意味着，KF1归一化平面上e1Pe2与极平面O1PO2是相似的，因为 e1P、e2P分别与O1P、O2P重叠，
            // 且e1、e2都在KF1的归一化平面上{事实上，像素平面可以认为它与归一化平面重叠，只是要使用焦距fx、fy缩放, cx、cy平移而已}
            // 那么, e1e2必然平行于O1O2, 那么极线方向我们自然可以写出来：
            ep1.x() = P12.x() * cam_->fx_; // 乘以焦距缩放到像素坐标
            ep1.y() = P12.y() * cam_->fy_;
            cout<< lk1 << "  " << "ep1-2: " << ep1.transpose() << endl;

        }
        ep1/=ep1.norm();
        // 这里我们使用双线性插值来获取光度，这样就不用担心四舍五入的问题了
        Eigen::Vector2d p1 = lk1->uv_;
        Eigen::Vector2d p1m1 = p1 - ep1, p1m2 = p1 - 2*ep1,
                        p1p1 = p1 + ep1, p1p2 = p1 + 2*ep1;
        //ep1 = {1, 1};
        //        Eigen::Vector2d p1m1 = p1 - ep1, p1m2 = p1 - 2*ep1,
        //                p1p1 = p1 + ep1, p1p2 = p1 + 2*ep1;
        vector<Eigen::Vector2d> debugPx1{p1, p1m1, p1m2, p1p1, p1p2};
        cout << "(p1p2-p1m2).norm: " << (p1p2-p1m2).norm() << endl;
        
        if(!InRange(grayImg_, p1m2.cast<int>()) || !InRange(grayImg_, p1p2.cast<int>())) {
            cout << "ERROR p1m2, p1p2: " << p1m2.transpose() << " | " << p1p2.transpose() << endl;
            return {};
        }
        // OK，接下来在对极线上等距取5个点，据此来计算SSD
        double v1[5];
        v1[2] = BilinearInterpolate<uchar>(grayImg_, p1),
            v1[1] = BilinearInterpolate<uchar>(grayImg_, p1m1),
            v1[0] = BilinearInterpolate<uchar>(grayImg_, p1m2),
            v1[3] = BilinearInterpolate<uchar>(grayImg_, p1p1),
            v1[4] = BilinearInterpolate<uchar>(grayImg_, p1p2);
        double s1 = 0;
        double avg1 = 0;
        for(int i = 0; i < 5; ++i) {
            s1 += v1[i];
        }
        avg1 = s1/5;

        auto px2IsEdge = [&edgeImg2] (const Eigen::Vector2d px2) -> bool{
            bool isEdge = edgeImg2.at<uchar>(px2.y(), px2.x()) == 0;
            if(!isEdge) {
                // 允许小的像素偏差
                const int ix = px2.x(), iy = px2.y();
                vector<Eigen::Vector2i> xy = {{0, 1}, {0, -1}, {-1, 0}, {1, 0}, {-1, 1}, {1, 1}, {-1, -1}, {1, -1}};
                for(const Eigen::Vector2i &dp : xy) {
                    Point2i p(ix+dp.x(), iy+dp.y());
                    if(edgeImg2.at<uchar>(p) == 0) {
                        isEdge = true;
                        break;
                    }
                }
            }
            return isEdge || 1;
        };



        double bestScore = DBL_MAX, secondBestScore = DBL_MAX;
        Eigen::Vector2d bestP2{1000, 1000}, secondBestP2{1000, 1000};

        Eigen::Vector2d &p2 = farPx2;
        cout << "farPx2, ep2: " << p2.transpose() << " | " << ep2.transpose() << endl;
        Eigen::Vector2d p2m1 = p2 - ep2, p2m2 = p2 - 2*ep2,
                        p2p1 = p2 + ep2, p2p2 = p2 + 2*ep2;
        vector<Eigen::Vector2d> debugPx2{p2, p2m1, p2m2, p2p1, p2p2};
        //double v2, v2m1, v2m2, v2p1, v2p2;
        double v2[5];
        //double s2 = 0, avg2 = 0; // 可以使用滑窗计算
        const Mat &img2 = kf2->grayImg_;
        while ((p2 - nearPx2).norm() > 1) {
            //cout << "p2m2, p2p2: " << p2m2.transpose() << " | " << p2p2.transpose() << endl;
            if (InRange(kf2->grayImg_, p2m2.cast<int>()) && InRange(kf2->grayImg_, p2p2.cast<int>()) && px2IsEdge(p2)) {
                    v2[2] = BilinearInterpolate<uchar>(img2, p2);
                    v2[1] = BilinearInterpolate<uchar>(img2, p2m1),
                    v2[0] = BilinearInterpolate<uchar>(img2, p2m2),
                    v2[3] = BilinearInterpolate<uchar>(img2, p2p1),
                    v2[4] = BilinearInterpolate<uchar>(img2, p2p2);

                    double s2 = 0, avg2 = 0;
                    for(int i = 0; i < 5; ++i) {
                        s2 += v2[i];
                    }
                    avg2 = s2/5;

                //cout << "v2m2, v2, v2p2: " << v2m2 << ", " << v2 << ", " << v2p2 << endl;
                const double score = CalculateSSD(v1, v2, avg1, avg2);
                
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

            p2m2 = p2m1,
            p2m1 = p2,
            p2 = p2p1,
            p2p1 = p2p2,
            p2p2 += ep2;
            debugPx2.push_back(p2p2);
            debugPx1.push_back(p1);
            p1 += ep1;
        }

        //if(InRange(edgeImg2, bestP2.cast<int>())) {
        //    cout << "best, second score mean: " << bestScore/5 << " " << secondBestScore/5 << endl;
        //    debugPx2.push_back(bestP2);

        //}
        //debugPx1.push_back(lk1->uv_);

        cout << "debugPx1, debugPx2 size: " << debugPx1.size() << ", " << debugPx2.size() << endl;
        cout << "lk1->uv_: " << lk1->uv_.transpose() << endl;
        cout << "p1m2, p1p2: " << p1m2.transpose() << " | " << p1p2.transpose() << endl;
        cout << "bestP2: " << bestP2.transpose() << endl;
        //DrawMatch(grayImg_, kf2->grayImg_, debugPx1, debugPx2, "current point 2 all Epipolar constraint matches", 1, 1000000);

        const bool smallScore = bestScore < config->maxDescriptorDist * 5;
        if(!smallScore) {
            return {};
        }
        const bool goodScore = bestScore< 1.0 * secondBestScore;
        const double badDist = (bestP2 - secondBestP2).norm() > 10;
        if(goodScore || (!badDist && !goodScore )) {
            return {bestP2};
        }

        return {};
    // }
}