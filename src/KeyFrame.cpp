#include "KeyFrame.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <opencv2/highgui.hpp>
#include <random>
#include <string>
#include <vector>

#include "Config.h"
#include "Eigen/src/Core/Matrix.h"
#include "Pose.h"
#include "Utils.h"

using namespace std;
using namespace cv;

#if defined(WRITE_MATCH_PAIR_IMAGE)
cv::VideoWriter KeyFrame::debugVideoWriter;
constexpr int kDrawMatchNumEachFrame = 100;
cv::VideoWriter KeyFrame::debugTriangulateWriter;
#endif

class Landmark;

KeyFrame::KeyFrame(const Mat& img, const Pose& Twc, std::shared_ptr<Camera> cam,
                   const int id, const int level)
    : id_(id),
      grayImg_(img),
      cam_(cam),
      Twc_{Twc},
      Tcw_(Twc.Inverse()),
      priorTwc_(Twc),
      level_(level) {
    edgeImg_.resize(level);
    dist_.resize(level);
    dx_.resize(level);
    dy_.resize(level);
    unPx_.resize(level);
}

KeyFrame::KeyFrame(const KeyFrame& f)
    : id_(f.id_),
      grayImg_(f.grayImg_),
      edgeImg_(f.edgeImg_),
      dist_(f.dist_),
      dx_(f.dx_),
      dy_(f.dy_),
      cam_(f.cam_),
      Twc_(f.Twc_),
      Tcw_(f.Tcw_),
      priorTwc_(f.priorTwc_),
      level_(f.level_),
      unPx_(f.unPx_),
      descriptor_(f.descriptor_),
      pointMapId_(f.pointMapId_),
      outOfRange_(f.outOfRange_),
      convergeEdgeNum_(f.convergeEdgeNum_) {
    // vector内的堆内存需要先释放
    // 不能这样子，这是构造函数，默认的内存应该是干净的，
    // 否则你应该调用赋值构造
    for (Landmark* lk : landmark_) {
        if (lk != nullptr) {
            delete lk;
        }
    }
    landmark_.clear();
    landmark_.reserve(f.landmark_.size());
    for (Landmark* lk : f.landmark_) {
        landmark_.push_back(new Landmark(*lk));
        // !!!Attention: 指针成员变量需要小心处理，因为如果其指向栈内存，由于栈内存会被系统回收，
        // 因此可能产生意外情况
        landmark_.back()->host_ = this;
    }
    debugGrayImg_ = f.debugGrayImg_;
    depthImage_ = f.depthImage_;
}

KeyFrame::~KeyFrame() {
    // 由于Landmar与KeyFrame相互引用，所以之前将析构函数放在头文件导致landmark_内存无法释放？？
    for (Landmark* lk : landmark_) {
        if (lk != nullptr) {
            delete lk;
            lk = nullptr;
        }
    }
    // ReleaseMat(); // 不需要手动释放
    if (invDepthUncertaintyFile_.is_open()) {
        invDepthUncertaintyFile_.close();
    }

    std::cout << this << " Releasw KF id: " << id_ << std::endl;
}

void KeyFrame::operator=(const KeyFrame& f) {
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
    for (Landmark* lk : landmark_) {
        if (lk != nullptr) {
            delete lk;
        }
    }
    landmark_.clear();
    landmark_.reserve(f.landmark_.size());
    for (Landmark* lk : f.landmark_) {
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
    depthImage_ = f.depthImage_;
}

void KeyFrame::CannyEdgeDetect() {
#ifdef Undistort
    static Mat map1, map2;

    if (map1.empty()) {
        Mat D;
        if (config->model == "fisheye") {
            D = (cv::Mat_<float>(4, 1) << cam_->k1_, cam_->k2_, cam_->k3_,
                 cam_->k4_);
        } else if (config->model == "pinhole") {
            D = (cv::Mat_<float>(5, 1) << cam_->k1_, cam_->k2_, cam_->k3_,
                 cam_->k4_, cam_->k5_);
        }
        Mat R = cv::Mat::eye(3, 3, CV_32F);
        // 去畸变后，可以使用原有的内参，也可以使用自己定义的新内参，rebvo就是自己定义了新的 zf=(fx+fy)*0.5
        Mat K = (cv::Mat_<float>(3, 3) << cam_->fx_, 0, cam_->cx_, 0, cam_->fy_,
                 cam_->cy_, 0, 0, 1);

        const double fm = (cam_->fx_ + cam_->fy_) * 0.5 * 1.1;
        Mat newK = (cv::Mat_<float>(3, 3) << fm, 0, cam_->cx_, 0, fm, cam_->cy_,
                    0, 0, 1);

        // 畸变模板应该只需要计算一次！！！
        cv::initUndistortRectifyMap(K, D, cv::Mat(), newK,
                                    cv::Size(grayImg_.cols, grayImg_.rows),
                                    CV_8UC1, map1, map2);
        cam_->UpdateIntrinsicParam(fm, fm);
    }

    cv::remap(grayImg_, grayImg_, map1, map2, cv::INTER_LINEAR);
#endif

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
#if 0
    cv::Mat blurred = grayImg_.clone();
#else
    Mat blurred = grayImg_.clone();
#endif
    // 应用高斯滤波来平滑边缘
    cv::GaussianBlur(grayImg_, blurred, cv::Size(3, 3), 1);

    debugGrayImg_ = grayImg_.clone();
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();

    const double lowerThreshold = config->cannyLowerTh;  // 下限阈值
    const double upperThreshold =
        config->cannyupperTh;  // 上限阈值，越小提取边缘越多
    int apertureSize = 3;      // 应用Sobel算子的窗口大小

    for (int lvl = 0; lvl < level_; ++lvl) {
        Mat blur_i;
        double ratio = 1. / pow(2, lvl);
        if (lvl > 0)
            cv::resize(blurred, blur_i,
                       cv::Size(ratio * blurred.cols, ratio * blurred.rows));
        else
            blur_i = blurred;

        Canny(blur_i, edgeImg_[lvl], lowerThreshold, upperThreshold,
              apertureSize);
        // 膨胀及腐蚀操作，看起来收益不大
        // cv::Mat er = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        //cv::Mat di = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(8, 8));
        // cv::erode(blur_i, blur_i, er);
        //cv::dilate(blur_i, blur_i, di);

        chrono::steady_clock::time_point t3 = chrono::steady_clock::now();

        if (config->messageLevel == MessageLevel::Debug)
            cout << "Extract canny edge spend "
                 << chrono::duration<double>(t3 - t2).count() << "s"
                 << " & Gaussian Blur spend "
                 << chrono::duration<double>(t2 - t1).count() << endl;

        vector<Eigen::Vector2i> px;
        // 取出边缘像素点
        // 这里跳过了6个图像边缘的像素
        constexpr int jump = 6;
        for (int x = jump; x < edgeImg_[lvl].cols - jump; ++x) {
            for (int y = jump; y < edgeImg_[lvl].rows - jump; ++y) {
                if (edgeImg_[lvl].at<uchar>(y, x) != 0 &&
                    (lvl != 0 || IsFastPoint(grayImg_, {x, y}))) {
                    px.push_back({x, y});
                }
            }
        }

        // cv::imshow("distor", edgeImg_[i]);
        // cv::waitKey(0);

        chrono::steady_clock::time_point t4 = chrono::steady_clock::now();
#ifdef Undistort
        unPx_[lvl].clear();
        for (const Eigen::Vector2i& p : px) {
            unPx_[lvl].push_back({p.x(), p.y()});
        }
#else
        unPx_[lvl] =
            cam_->UndistortPoints(px, lvl);  // 去畸变后的像素平面上的点
#endif
        chrono::steady_clock::time_point t5 = chrono::steady_clock::now();
        if (config->messageLevel == MessageLevel::Debug)
            cout << "Undistort " << px.size() << "points spend "
                 << chrono::duration<double>(t5 - t4).count() << "s" << endl;

        edgeImg_[lvl] =
            Mat::ones(edgeImg_[lvl].rows, edgeImg_[lvl].cols, CV_8UC1) * 255;
        vector<Eigen::Vector2i>::iterator it = unPx_[lvl].begin();
        int id = -1, edgeNum = 0;
        while (it != unPx_[lvl].end()) {
            ++id;
            const Eigen::Vector2i& p = *it;
            // 边缘点置为黑色
            if (InRange(edgeImg_[lvl], {p.x(), p.y()})) {
                edgeImg_[lvl].at<uchar>(int(p.y()), int(p.x())) = 0;
                if (lvl == 0) {
                    debugGrayImg_.at<uchar>(int(p.y()), int(p.x())) = 255;
                }
                // 使用未去畸变像素邻域
                const int x = px[id].x(), y = px[id].y();

                // 计算描述子
                // const Mat &m = grayImg_;
                // Eigen::Matrix<float, kDescriptorPatchSize, 1> d;
                // d << m.at<uchar>(y-1, x-1), m.at<uchar>(y-1, x), m.at<uchar>(y-1, x+1),
                //      m.at<uchar>(y, x-1), m.at<uchar>(y, x), m.at<uchar>(y, x+1),
                //      m.at<uchar>(y+1, x-1), m.at<uchar>(y+1, x), m.at<uchar>(y+1, x+1);
                // descriptor_.emplace_back(d);

                // 只保留第0层金字塔
                if (lvl == 0) {
                    const int w = grayImg_.cols, h = grayImg_.rows;
                    //descriptor_.push_back(::CalculateDescriptor(grayImg_, {x, y}) );
#if USE_POINT_MAP_ID
                    pointMapId_.insert({{p.x(), p.y()}, edgeNum++});
#else
                    pointMapId_.insert({p.y() * w + p.x(), edgeNum++});
#endif
                }
                ++it;
            } else {
                it = unPx_[lvl].erase(it);
            }
        }

        //cv::imshow("undistor", edgeImg_[i]);
        //cv::waitKey(0);
    }
}

void KeyFrame::ExtractEdge() {
#ifdef Undistort
    static Mat map1, map2;
    if (map1.empty()) {
        Mat D;
        if (config->model == "fisheye") {
            D = (cv::Mat_<float>(4, 1) << cam_->k1_, cam_->k2_, cam_->k3_,
                 cam_->k4_);
        } else if (config->model == "pinhole") {
            D = (cv::Mat_<float>(5, 1) << cam_->k1_, cam_->k2_, cam_->k3_,
                 cam_->k4_, cam_->k5_);
        }
        Mat R = cv::Mat::eye(3, 3, CV_32F);
        // 去畸变后，可以使用原有的内参，也可以使用自己定义的新内参，rebvo就是自己定义了新的 zf=(fx+fy)*0.5
        Mat K = (cv::Mat_<float>(3, 3) << cam_->fx_, 0, cam_->cx_, 0, cam_->fy_,
                 cam_->cy_, 0, 0, 1);

        // 使用新的内参投影，避免出现黑色的部分
        double newFx = (cam_->fx_ + cam_->fy_) * 0.5 * 1.1;
        Mat newK = (cv::Mat_<float>(3, 3) << newFx, 0, cam_->cx_, 0, newFx,
                    cam_->cy_, 0, 0, 1);

        // 畸变模板应该只需要计算一次！！！
        cv::initUndistortRectifyMap(K, D, cv::Mat(), newK,
                                    cv::Size(grayImg_.cols, grayImg_.rows),
                                    CV_8UC1, map1, map2);

        cam_->UpdateIntrinsicParam(newFx, newFx);
    }
    cv::remap(grayImg_, grayImg_, map1, map2, cv::INTER_LINEAR);
#endif

#define USE_CANNY 0
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
#if USE_CANNY
    cv::Mat blurred = grayImg_.clone();
    cv::GaussianBlur(grayImg_, blurred, cv::Size(5, 5), 1);
#else
    Mat blurred = grayImg_;
#endif

    debugGrayImg_ = grayImg_.clone();
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();

    const double lowerThreshold = config->cannyLowerTh;  // 下限阈值
    const double upperThreshold =
        config->cannyupperTh;  // 上限阈值，越小提取边缘越多
    int apertureSize = 3;      // 应用Sobel算子的窗口大小

    for (int lvl = 0; lvl < level_; ++lvl) {
        Mat blur_i;
        double ratio = 1. / pow(2, lvl);
        if (lvl > 0)
            cv::resize(blurred, blur_i,
                       cv::Size(ratio * blurred.cols, ratio * blurred.rows));
        else
            blur_i = blurred;

#if USE_CANNY
        Canny(blur_i, edgeImg_[lvl], lowerThreshold, upperThreshold,
              apertureSize, true);

        vector<Eigen::Vector2i> edgePx;
        constexpr int jump = 6;
        for (int x = jump; x < edgeImg_[lvl].cols - jump; ++x) {
            for (int y = jump; y < edgeImg_[lvl].rows - jump; ++y) {
                if (edgeImg_[lvl].at<uchar>(y, x) == 255) {
                    edgePx.push_back({x, y});
                }
            }
        }

        edgeImg_[lvl] =
            Mat::ones(edgeImg_[lvl].rows, edgeImg_[lvl].cols, CV_8UC1) * 255;
        for (const Eigen::Vector2i& p : edgePx) {
            edgeImg_[lvl].ptr<uchar>(p.y())[p.x()] = 0;
            if (lvl == 0) {
                debugGrayImg_.at<uchar>(p.y(), p.x()) = 255;
            }
        }
#else
        Mat dx, dy;
        CaculateDerivative<uchar>(blur_i, dx, dy);
        edgeImg_[lvl] = Mat::ones(blur_i.rows, blur_i.cols, CV_8UC1) * 255;
        int w = blur_i.cols, h = blur_i.rows;
        constexpr int jump = 6;
        float *datax = dx.ptr<float>(), *datay = dy.ptr<float>();
        uchar *dataEdge = edgeImg_[lvl].data, *dataDebug = debugGrayImg_.data;

        const int minGrant = config->minGradientOFkeyPoint;
        for (int i = jump; i < h - jump; ++i)
            for (int j = jump; j < w - jump; ++j) {
                const int id = i * w + j;
                if (abs(datax[id]) + abs(datay[id]) > minGrant) {
                    dataEdge[id] = 0;
                    if (lvl == 0) {
                        dataDebug[id] = 255;
                    }
                }
            }

            // if(lvl==0)
            //     DrawPerpendicularAndParallelDirectionOFedge(edgeImg_[0], dx, dy);
#endif

#ifndef Undistort
        // edgePx = cam_->UndistortPoints(edgePx, lvl);
#endif
    }
    chrono::steady_clock::time_point t3 = chrono::steady_clock::now();

    if (config->messageLevel == MessageLevel::Debug || 1)
        cout << "Extract canny edge spend "
             << chrono::duration<double>(t3 - t2).count() << "s"
             << " & Gaussian Blur spend "
             << chrono::duration<double>(t2 - t1).count() << endl;
}

void KeyFrame::GenerateDTandDerivative() {
    for (int lvl = 0; lvl < level_; ++lvl) {
        dist_[lvl] = GetDistanceTransform(edgeImg_[lvl]);
        CaculateDerivative<float>(dist_[lvl], dx_[lvl], dy_[lvl]);
    }

    // 归一化显示梯度结果
    // cv::Mat showGx = cv::abs(dx_[0]),
    //    showGy = cv::abs(dy_[0]);
    // cv::normalize(showGx, showGx, 0, 255, cv::NORM_MINMAX);
    // cv::normalize(showGy, showGy, 0, 255, cv::NORM_MINMAX);
    // showGx.convertTo(showGx, CV_8U);
    // showGy.convertTo(showGy, CV_8U);
    // cv::imshow("showGx", showGx);
    // cv::imshow("showGy", showGy);
    // cv::waitKey(0);
}

void KeyFrame::GenerateKeyPoint() {
    const float maxDist = config->maxDTdistOFkeyPoint;
    int count = 0;
    const int w = dist_[0].cols, h = dist_[0].rows;
    for (int lvl = 0; lvl < level_; ++lvl)
        for (int i = 0; i < dist_[0].rows; ++i) {
            for (int j = 0; j < dist_[0].cols; ++j) {
                if (dist_[0].at<float>(i, j) <= maxDist) {
                    // 插入关键点
                    unPx_[lvl].push_back({j, i});
                    if (lvl == 0) {
#if USE_POINT_MAP_ID
                        pointMapId_.insert({{j, i}, count++});
#else
                        pointMapId_.insert({i * w + j, count++});
#endif
                    }
                }
            }
        }
}

void KeyFrame::AddReportElement(const std::string& key) {
    if (matchResultStatiscs_.count(key)) {
        matchResultStatiscs_[key]++;
    } else {
        matchResultStatiscs_[key] = 1;
    }
}

void KeyFrame::ResetDebugMessage() {
    matchResultStatiscs_.clear();
}

void KeyFrame::ReportMatchResult() {
    cout << "keyframe id: " << id_ << "Match result statiscs report: " << endl;
    int sum = 0;
    for (const auto& p : matchResultStatiscs_) {
        sum += p.second;
    }

    for (const auto& p : matchResultStatiscs_) {
        const float ratio = static_cast<float>(p.second) / sum;
        cout << p.first << ": num=" << p.second << ", ratio: " << ratio << endl;
    }
    cout << endl << endl;
}

#if defined(WRITE_MATCH_PAIR_IMAGE)
void KeyFrame::DrawBestMatchEachFrame(const Eigen::Vector2i& kp1,
                                      const Eigen::Vector2i& matchKp2,
                                      const cv::Mat& debugImg2) {
    if (videoBestMatchDebugImg_.empty()) {
        videoBestMatchDebugImg_ =
            cv::Mat(debugGrayImg_.rows, debugGrayImg_.cols * 2, CV_8UC3,
                    cv::Scalar{0, 0, 0});
        Mat im1, im2;
        cvtColor(debugGrayImg_, im1, COLOR_GRAY2BGR);
        cvtColor(debugImg2, im2, COLOR_GRAY2BGR);
        im1.copyTo(videoBestMatchDebugImg_.colRange(0, debugGrayImg_.cols));
        im2.copyTo(videoBestMatchDebugImg_.colRange(
            debugGrayImg_.cols, videoBestMatchDebugImg_.cols));
    }

    const int colorId = abs(rand()) % kColor.size();
    int idx = -1;
    cv::Vec3b color(0, 0, 0);
    for (const auto& c : kColor) {
        if (++idx == colorId) {
            color = c.second;
            break;
        }
    }
    int radius = 1;

    cv::Point p1(kp1.x(), kp1.y());
    cv::circle(videoBestMatchDebugImg_, p1, radius, color, 1);
    cv::Point bestMatchP2 =
        cv::Point(debugGrayImg_.cols + matchKp2.x(), matchKp2.y());
    cv::circle(videoBestMatchDebugImg_, bestMatchP2, radius, color, 1);
    cv::line(videoBestMatchDebugImg_, p1, bestMatchP2, color, 1);
}

void KeyFrame::WriteDebugImage2VideoEachFrame(const int kf2Id) {
    if (videoBestMatchDebugImg_.empty()) {
        return;
    }
    int start_text_row = 20;
    int step_text_row = 10;
    cv::putText(videoBestMatchDebugImg_,
                "match point keyframe2 id: " + to_string(kf2Id),
                cv::Point(10, (start_text_row += step_text_row)),
                cv::FONT_ITALIC, 1.0, kColor.at("red"), 1);
    cv::putText(videoEpipolarMatchDebugImg_,
                "match ep keyframe2 id: " + to_string(kf2Id),
                cv::Point(10, start_text_row), cv::FONT_ITALIC, 1.0,
                kColor.at("red"), 1);
    if (!videoEpipolarFailMatchDebugImg_.empty()) {
        cv::putText(videoEpipolarFailMatchDebugImg_,
                    "fail ep keyframe2 id: " + to_string(kf2Id),
                    cv::Point(10, start_text_row), cv::FONT_ITALIC, 1.0,
                    kColor.at("red"), 1);
    }

    // 初始化边缘匹配的debug视频写入器
    if (!KeyFrame::debugVideoWriter.isOpened()) {
        string videoPath = config->debugMessageSaveFolder;
        const string debugVideoName("edge_host_match_cur_frame_video.avi");
        videoPath += "/" + debugVideoName;
        // 或者使用未压缩的格式（如果磁盘IO不是瓶颈）
        int fourcc = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
        int fps = 30;
        KeyFrame::debugVideoWriter.open(videoPath, fourcc, fps,
                                        videoBestMatchDebugImg_.size(), true);
        if (!KeyFrame::debugVideoWriter.isOpened()) {
            cerr << "Open debug video path: " << videoPath << " failed" << endl;
            exit(-1);
        }
        cout << "Open debug video path: " << videoPath << endl;
    }

    debugVideoWriter.write(videoEpipolarMatchDebugImg_);
    debugVideoWriter.write(videoBestMatchDebugImg_);
    debugVideoWriter.write(videoEpipolarFailMatchDebugImg_);
    videoEpipolarMatchDebugImg_.release();
    videoBestMatchDebugImg_.release();
    videoEpipolarFailMatchDebugImg_.release();
}

void KeyFrame::WriteDebugTriangulateCase2Video() {

    // 初始化边缘匹配的debug视频写入器
    string videoPath = config->debugMessageSaveFolder;

    for (auto& nameMapImgs : triPointMapDebugImage_) {
        const string pointName = nameMapImgs.first;
        const string curVideoPath =
            fmt::format("{}/kfId_{}_{}.avi", videoPath, id_, pointName);
        int fourcc = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
        int fps = 30;
        KeyFrame::debugTriangulateWriter.open(
            curVideoPath, fourcc, fps, nameMapImgs.second[0].size(), true);
        if (!KeyFrame::debugTriangulateWriter.isOpened()) {
            cerr << "Open debug video path: " << curVideoPath << " failed"
                 << endl;
            //exit(-1);
            return;
        }
        cout << "Open debug video path: " << curVideoPath << endl;

        for (cv::Mat& img : nameMapImgs.second) {
            debugTriangulateWriter.write(img);
            img.release();
        }

        KeyFrame::debugTriangulateWriter.release();
    }
}

void KeyFrame::DrawEpipolarMatchEachFrame(const Eigen::Vector2i& kp1,
                                          const Eigen::Vector2i& lp2Start,
                                          const Eigen::Vector2i& lp2End,
                                          const Eigen::Vector2i& matchKp2,
                                          const cv::Mat& debugImg2) {
    if (videoEpipolarMatchDebugImg_.empty()) {
        videoEpipolarMatchDebugImg_ =
            cv::Mat(debugGrayImg_.rows, debugGrayImg_.cols * 2, CV_8UC3,
                    cv::Scalar{0, 0, 0});
        Mat im1, im2;
        cvtColor(debugGrayImg_, im1, COLOR_GRAY2BGR);
        cvtColor(debugImg2, im2, COLOR_GRAY2BGR);
        im1.copyTo(videoEpipolarMatchDebugImg_.colRange(0, debugGrayImg_.cols));
        im2.copyTo(videoEpipolarMatchDebugImg_.colRange(
            debugGrayImg_.cols, videoEpipolarMatchDebugImg_.cols));
    }

    const int colorId = abs(rand()) % kColor.size();
    int idx = -1;
    cv::Vec3b matchColor(0, 0, 0);
    for (const auto& c : kColor) {
        if (++idx == colorId) {
            matchColor = c.second;
            break;
        }
    }
    int radius = 2;

    cv::Point p1(kp1.x(), kp1.y());
    cv::circle(videoEpipolarMatchDebugImg_, p1, radius, matchColor, 1);
    cv::Point bestMatchP2 =
        cv::Point(debugGrayImg_.cols + matchKp2.x(), matchKp2.y());
    cv::circle(videoEpipolarMatchDebugImg_, bestMatchP2, radius, matchColor, 1);
    cv::line(videoEpipolarMatchDebugImg_, p1, bestMatchP2, matchColor, 1);

    // 画极线起终点，起点绿色，终点红色，连线蓝色
    const cv::Point lp1(debugGrayImg_.cols + lp2Start.x(), lp2Start.y());
    const cv::Point lp2(debugGrayImg_.cols + lp2End.x(), lp2End.y());
    cv::circle(videoEpipolarMatchDebugImg_, lp1, radius, kColor.at("green"), 1);
    cv::circle(videoEpipolarMatchDebugImg_, lp2, radius, kColor.at("red"), 1);
    cv::line(videoEpipolarMatchDebugImg_, lp1, lp2, kColor.at("blue"), 1);
}

void KeyFrame::DrawFailEpipolarMatchEachFrame(const Eigen::Vector2i& kp1,
                                              const Eigen::Vector2i& lp2Start,
                                              const Eigen::Vector2i& lp2End,
                                              const cv::Mat& debugImg2) {
    if (videoEpipolarFailMatchDebugImg_.empty()) {
        videoEpipolarFailMatchDebugImg_ =
            cv::Mat(debugGrayImg_.rows, debugGrayImg_.cols * 2, CV_8UC3,
                    cv::Scalar{0, 0, 0});
        Mat im1, im2;
        cvtColor(debugGrayImg_, im1, COLOR_GRAY2BGR);
        cvtColor(debugImg2, im2, COLOR_GRAY2BGR);
        im1.copyTo(
            videoEpipolarFailMatchDebugImg_.colRange(0, debugGrayImg_.cols));
        im2.copyTo(videoEpipolarFailMatchDebugImg_.colRange(
            debugGrayImg_.cols, videoEpipolarFailMatchDebugImg_.cols));
    }

    const int colorId = abs(rand()) % kColor.size();
    int idx = -1;
    cv::Vec3b matchColor(0, 0, 0);
    for (const auto& c : kColor) {
        if (++idx == colorId) {
            matchColor = c.second;
            break;
        }
    }
    int radius = 2;

    cv::Point p1(kp1.x(), kp1.y());
    cv::circle(videoEpipolarFailMatchDebugImg_, p1, radius, matchColor, 1);

    // 画极线起终点，起点绿色，终点红色，连线蓝色
    const cv::Point lp1(debugGrayImg_.cols + lp2Start.x(), lp2Start.y());
    const cv::Point lp2(debugGrayImg_.cols + lp2End.x(), lp2End.y());
    cv::circle(videoEpipolarFailMatchDebugImg_, lp1, radius, kColor.at("green"),
               1);
    cv::circle(videoEpipolarFailMatchDebugImg_, lp2, radius, kColor.at("red"),
               1);
    cv::line(videoEpipolarFailMatchDebugImg_, lp1, lp2, kColor.at("blue"), 1);

    cv::line(videoEpipolarFailMatchDebugImg_, p1, lp1, kColor.at("green"), 1);
}

void KeyFrame::DrawTriangulateCase(
    const double estD1, const double estD2, const Landmark& lk1,
    const Eigen::Vector2i& epipolarP1, const Eigen::Vector2i& matchKp2,
    const Eigen::Vector2i& farPx2, const Eigen::Vector2i& nearPx2,
    const cv::Mat& debugImg2, const Pose& T12, const bool success) {
    cv::Mat showImg(debugGrayImg_.rows, debugGrayImg_.cols * 2, CV_8UC3,
                    cv::Scalar{0, 0, 0});
    Mat im1, im2;
    cvtColor(debugGrayImg_, im1, COLOR_GRAY2BGR);
    cvtColor(debugImg2, im2, COLOR_GRAY2BGR);
    im1.copyTo(showImg.colRange(0, debugGrayImg_.cols));
    im2.copyTo(showImg.colRange(debugGrayImg_.cols, showImg.cols));

    int start_text_row = 20;
    int step_text_row = 20;
    cv::putText(showImg, T12.QwbString(), cv::Point(10, (start_text_row)),
                cv::FONT_ITALIC, 0.8, kColor.at("red"), 1);
    cv::putText(showImg, T12.PwbString(),
                cv::Point(10, (start_text_row += step_text_row)),
                cv::FONT_ITALIC, 0.8, kColor.at("red"), 1);
    const string caseName = success ? "Suc tri" : "Fai tri";
    cv::putText(showImg, caseName,
                cv::Point(10, (start_text_row += step_text_row)),
                cv::FONT_ITALIC, 0.8, kColor.at("red"), 1);

    const cv::Vec3b& matchColor = kColor.at("yellow");

    int radius = 3;

    cv::Vec3b nearColor(0, 255, 0);
    cv::Vec3b farColor(0, 0, 255);
    const cv::Point pointDiff(debugGrayImg_.cols, 0);
    const Eigen::Vector2i& kp1 = lk1.uv_;
    cv::Point p1(kp1.x(), kp1.y());
    cv::Point p2(matchKp2.x(), matchKp2.y());
    constexpr double kTextRatio = 0.5;
    const cv::Point textDiff(5, 0);
    // 写必要信息
    cv::putText(showImg,
                fmt::format("({}, {}, {:.1f}, {:.1f}, {})", p1.x, p1.y, estD1,
                            lk1.trueDepth_, lk1.obvTime_ + 1),
                p1 + textDiff, cv::FONT_ITALIC, kTextRatio, kColor.at("red"),
                1);
    cv::putText(showImg, fmt::format("({}, {}, {:.1f})", p2.x, p2.y, estD2),
                p2 + textDiff + pointDiff, cv::FONT_ITALIC, kTextRatio,
                kColor.at("red"), 1);
    // 画极线起终点，起点绿色，终点红色，连线蓝色
    cv::line(showImg, p1, p2 + pointDiff, matchColor, 1);

    // 画极线以查看匹配是否准确
    cv::Point ep1(epipolarP1.x(), epipolarP1.y());
    cv::line(showImg, p1, ep1, kColor.at("white"), 1);
    cv::circle(showImg, p1, radius, matchColor, 1);
    cv::circle(showImg, ep1, 2, nearColor, -1);

    cv::Point n2(nearPx2.x(), nearPx2.y());
    cv::Point f2(farPx2.x(), farPx2.y());
    cv::line(showImg, n2 + pointDiff, f2 + pointDiff, kColor.at("white"), 1);
    cv::circle(showImg, p2 + pointDiff, radius, matchColor, 1);
    cv::circle(showImg, n2 + pointDiff, 2, nearColor, -1);
    cv::circle(showImg, f2 + pointDiff, 2, farColor, -1);

    const string debugImgName = fmt::format("{}_{}", lk1.uv_.x(), lk1.uv_.y());
    triPointMapDebugImage_[debugImgName].emplace_back(showImg);
}
#endif

size_t KeyFrame::GenerateLandmark(
    KeyFrame& kf2, vector<vector<Eigen::Vector2d> >& debugGoodKp1,
    vector<vector<Eigen::Vector2d> >& debugGoodKp2, const int equalparts) {

    const Pose T21 = kf2.Tcw_ * Twc_;
    const Pose T12 = T21.Inverse();
    cout << "T21: " << T21 << endl;
    cout << "T12: " << T12 << endl;
    debugGoodKp1.resize(equalparts);
    debugGoodKp2.resize(equalparts);
    for (int i = 0; i < unPx_[0].size(); ++i) {
        const Eigen::Vector2i& upx = unPx_[0][i];
        landmark_.push_back(
            new Landmark(upx, this, cam_, descriptor_[i], kInitInvDepth));
        landmark_.back()->descriptor_ = descriptor_[i];
        // landmark_.back()->UpdateUncertainty();
    }

    return landmark_.size();
}

size_t KeyFrame::InitializeLandmark() {
    if (landmark_.empty()) {
        // initialKF会有该种情况
        landmark_.resize(unPx_[0].size(), nullptr);
    }

    const int w = grayImg_.cols;
    const int h = grayImg_.rows;

    for (int i = 0; i < static_cast<int>(unPx_[0].size()); ++i) {
        if (landmark_[i] != nullptr && !config->useDepthImage &&
            !config->debugWithTrueDepthImage) {
            continue;
        }
        const int x = unPx_[0][i].x();
        const int y = unPx_[0][i].y();
        // shared_ptr<KeyFrame>(this)会导致多源智能指针，它会释放多次KeyFrame导致报错
        // 若需要使用智能指针，必须保证this在此前已经由一个智能指针管理，然后使用shared_from_this()来获取，否则只能使用原始指针
        // landmark_.push_back(make_shared<Landmark>(upx, make_shared<KeyFrame>(this), cam_, 1.0) ); [ERROR double free]

        const uint64_t descriptor = 0;
        if (!config->useDepthImage && !config->debugWithTrueDepthImage &&
            depthImage_.empty()) {
            landmark_[i] = new Landmark(unPx_[0][i], this, cam_, descriptor,
                                        kInitInvDepth);
        } else {
            // double d = static_cast<double>(depthImage_.ptr<ushort>(y)[x]);
            double d = static_cast<double>(depthImage_.at<ushort>(y, x));
            if (d == 0) {
                landmark_[i] = new Landmark(unPx_[0][i], this, cam_, descriptor,
                                            kInitInvDepth);

            } else {
                d /= config->depthFactor;
                landmark_[i] = new Landmark(unPx_[0][i], this, cam_, descriptor,
                                            kInitInvDepth);
                landmark_[i]->trueDepth_ = 1.0 / d;
                if (config->useDepthImage) {
                    landmark_[i]->invDepthCov_ = 0.005;
                    landmark_[i]->obvTime_ = 1e3;
                    landmark_[i]->UpdateUncertainty(true);
                    landmark_[i]->invZ_ = 1.0 / d;
                }
            }
            // host帧也要增加与landmark的相互观测
            landmark_[i]->target_.insert({this, unPx_[0][i]});
        }
    }

    return landmark_.size();
}

double KeyFrame::UpdateDepth(const KeyFrame& kf2, int& findMatchNum) {

    double matchEdgeNum = 0;
    convergeEdgeNum_ = 0;  // 重新统计当前KF的收敛边缘点集
    findMatchNum = 0;

    const Pose Tc1c2 = priorTwc_.Inverse() * kf2.priorTwc_;
    const Pose Tc2c1 = Tc1c2.Inverse();
    if (Tc1c2.t_wb_.norm() < 1e-3 || config->useDepthImage) {
        // 位移过小，不能进行更新
        findMatchNum = landmark_.size();
        return 1.0;
    }

    // vector<Pose> vTwc{ Pose(), Tc1c2};
    // vector<Mat> imgs{ debugGrayImg_, kf2.debugGrayImg_ };
    // ShowCameraCone(vTwc, imgs, *cam_);

#if defined(WRITE_MATCH_PAIR_IMAGE)
    int drawCount = 0;
    // 使用随机设备对地图点进行乱序，避免debug图像生成的点对聚在一堆
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(landmark_.begin(), landmark_.end(), gen);
#endif

    ResetDebugMessage();
    for (int i = 0; i < static_cast<int>(landmark_.size()); ++i) {
        if (landmark_[i] == nullptr || landmark_[i]->IsOutOfRange()) {
            continue;
        }
        Landmark* lk1 = landmark_[i];
        lk1->lastFrameMatchPx_.setZero();  // 重设匹配点
        if (lk1->Converge()) {
            convergeEdgeNum_ += 1;
        }

        // 每个Landmark只能由一个host控制，在转移控制权之前，只能更新其在host系下的depth
        // const vector<Eigen::Vector2d> kp2 = lk1->FindMatches(kf2);
        Eigen::Vector2d bestPx2(0, 0), farPx2(0, 0), nearPx2(0, 0),
            epipolarP1(0, 0), ep2(0, 0);
        // 这里才是开始找匹配像素点
        const double error = FindMatchesWithEpipolarConstraintOnImagePlane(
            &kf2, lk1, bestPx2, farPx2, nearPx2, epipolarP1, ep2);

        if (lk1->obvTime_ == 0) {
            // 首次创建深度假设
            if (error == EpipolarMatchType::repeatTextureORbadDepth ||
                error == EpipolarMatchType::occulsionORnoBestMatch) {
                // 深度异常
                ++lk1->failObvTime_;

                // 由于是小基线，当最小视差角设的大时，这里也应该增大，或者不设置outOFrange
                // 因为已经由最小视差角及先验std约束保证了初始化的准确性
                if ((lk1->failObvTime_ >
                     config->maxFailObvTimeBeforeCreateDepth) &&
                    Tc1c2.t_wb_.norm() > 0.02) {
                    // 可以在后续视角好的时候，过程完成初始化
                    // 三角化精度与准确度存在矛盾，因此，当基线过大时，宁愿不生成深度，也不愿生成错误深度
                    lk1->SetOutOfRange();
                }

                continue;
            }
        }

        constexpr double varianceExpand[2] = {1.05, 1.1};  // {1.01, 1.1};

        // 深度量测更新
        if (error == EpipolarMatchType::outOFboundaryORabnormalDepth) {
            continue;

        } else if (error == EpipolarMatchType::repeatTextureORbadDepth) {
            // ++lk1->failObvTime_;
            lk1->invDepthCov_ *= varianceExpand[1];
            lk1->UpdateUncertainty(false);
            continue;

        } else if (error == EpipolarMatchType::occulsionORnoBestMatch) {
            lk1->invDepthCov_ *= varianceExpand[0];
            lk1->UpdateUncertainty(false);
            continue;

        } else if (!bestPx2.isApprox(Eigen::Vector2d::Zero())) {
            double estInvD1 = -1;
            double estInvD2 = -1;
            if (!GetHostAndCurFrameObservationDepth(
                    lk1->uv_.cast<double>(), bestPx2, cam_->Kinv_[0], Tc1c2,
                    estInvD1, estInvD2)) {
                cout << "2th Calculate d1: " << estInvD1 << " d2: " << estInvD2
                     << " failed! no update!"
                     << "kp1: " << lk1->uv_.transpose()
                     << " bestPx2: " << bestPx2.transpose() << endl;
                // ++lk1->failObvTime_;

#if defined(WRITE_MATCH_PAIR_IMAGE)
                if (lk1->IsDebugPoint()) {
                    DrawTriangulateCase(1.0 / estInvD1, 1.0 / estInvD2, *lk1,
                                        epipolarP1.cast<int>(),
                                        bestPx2.cast<int>(), farPx2.cast<int>(),
                                        nearPx2.cast<int>(), kf2.debugGrayImg_,
                                        Tc1c2, false);
                }

#endif
                continue;
            }

#if defined(WRITE_MATCH_PAIR_IMAGE)
            if (lk1->IsDebugPoint()) {
                DrawTriangulateCase(1.0 / estInvD1, 1.0 / estInvD2, *lk1,
                                    epipolarP1.cast<int>(), bestPx2.cast<int>(),
                                    farPx2.cast<int>(), nearPx2.cast<int>(),
                                    kf2.debugGrayImg_, Tc1c2, true);
            }
#endif
            // const double estCov =
            //     CalculateVariance(estInvD1, lk1->uv_.cast<double>(), bestPx2,
            //                       Tc2c1, cam_->Kinv_[0], cam_->K_[0]);
            const double estCov = CalculateVarianceByOffsetPx2(
                lk1->uv_.cast<double>(), bestPx2, ep2, cam_->Kinv_[0], Tc1c2,
                config->matchNoise);
            if (i % 1000 == 0)
                cout << "1th invz: " << lk1->invZ_
                     << ", cov: " << lk1->invDepthCov_
                     << ", bestInvDepth: " << estInvD1 << ", estCov: " << estCov
                     << endl;

            if (lk1->ObvUpdate(estInvD1, estCov)) {
                ++findMatchNum;
                lk1->lastFrameMatchPx_ =
                    bestPx2;  // 记录当前匹配成功点，后续用于异常深度剔除
            }
        }

        // 三角化后的深度异常值过大
        if (lk1->invDepthCov_ > config->maxObvDepthStd) {
            lk1->SetOutOfRange();
        }

        if (firstWriteUncertainty_) {
            invDepthUncertaintyFile_.open(
                fmt::format("{}/kf_{}_depth_uncertainty.csv",
                            config->debugMessageSaveFolder, id_));
            invDepthUncertaintyFile_
                << "#pointId, cov, invDepth, depth, "
                   "trueDepth, depthDiff, obvTime, failObvTime"
                << endl;
            firstWriteUncertainty_ = false;
        }
        // unf << " [" << to_string(lk1->depthRange_[0]) << ", " << to_string(lk1->depthRange_[1]) << "] std, depth: "
        if (lk1->trueDepth_ > 0.01 && lk1->AbnormalConvergeLandmark()) {
            const double estDepth = GetPositiveDepth(lk1->invZ_);
            invDepthUncertaintyFile_
                << lk1->uv_.x() << "_" << lk1->uv_.y() << ", "
                << lk1->invDepthCov_ << ", " << lk1->invZ_ << ", " << estDepth
                << ", " << lk1->trueDepth_ << ", "
                << (estDepth - lk1->trueDepth_) << ", " << lk1->obvTime_ << ", "
                << lk1->failObvTime_ << endl;
        }

#if defined(WRITE_MATCH_PAIR_IMAGE)
        if (++drawCount % kDrawMatchNumEachFrame == 0) {
            // 将各个最优匹配写入视频
            WriteDebugImage2VideoEachFrame(kf2.id_);
            drawCount = 0;
        }
#endif
    }

#if defined(WRITE_MATCH_PAIR_IMAGE)
    // 将各个最优匹配写入视频
    WriteDebugImage2VideoEachFrame(kf2.id_);
    drawCount = 0;
#endif

    ReportMatchResult();

    ++updateFrameCount_;

    if (config->messageLevel <= MessageLevel::Error)
        cout << setprecision(5)
             << "matchEdgeNum, convergeEdgeNum_: " << matchEdgeNum << " "
             << convergeEdgeNum_ << endl;
    cout << "successful findMatchNum: " << findMatchNum << endl;
    return double(findMatchNum) / landmark_.size();
}

// 使用极线约束跟踪每一个边缘点
int KeyFrame::TrackLandmarkByEpilorLine(const KeyFrame& kf1) {
    int trackNum = 0;

    const Pose Tc1c2 = kf1.priorTwc_.Inverse() * priorTwc_;
    if (Tc1c2.t_wb_.norm() < 0.01) {
        // 位移过小，不能进行更新
        return 1;
    }

    const Pose T21 = Tcw_ * kf1.Twc_;

    // 给新的KF2预分配内存
    if (landmark_.empty()) {
        landmark_ = vector<Landmark*>(unPx_[0].size(), nullptr);
    }

    int epipolarSearchNoneNum = 0;  // debug参数
    int depthOutofRangeNum = 0;

    const vector<Landmark*>& landmark = kf1.landmark_;
    for (int i = 0; i < landmark.size(); ++i) {
        if (landmark[i] == nullptr || landmark[i]->IsOutOfRange()) {
            // 未收敛的landmark也当作可追踪的，因其在未来可能收敛
            continue;
        }
        Landmark* lp1 = landmark[i];
        const vector<Eigen::Vector2d> kp2 = lp1->FindMatches(*this);
        if (kp2.empty()) {
            ++epipolarSearchNoneNum;
            continue;
        }

        for (const Eigen::Vector2d& p : kp2) {
            const int w = grayImg_.cols, h = grayImg_.rows;
            const Eigen::Vector3d pc1 =
                Triangulate(lp1->uv_.cast<double>(), p, T21, *cam_);
            if (pc1.z() >= lp1->depthRange_[0] &&
                pc1.z() <= lp1->depthRange_[1]) {
#if USE_POINT_MAP_ID
                const int vecId = pointMapId_[p.cast<int>()];
#else
                const int x = p.x(), y = p.y();
                const int vecId = pointMapId_[y * w + x];
#endif
                if (landmark_[vecId] != nullptr) {
                    // TODO: 选一个更好的，或者按照先来后到
                    break;
                }
                const uint64_t d2 = descriptor_[vecId];
                const uint64_t d1 = lp1->descriptor_;
                uint64_t score = CalculateDescriptorScore(d1, d2);
                if (score < config->goodDescriptorDist) {
                    // 增加相互观测
                    landmark_[vecId] = lp1;
                    lp1->target_.insert({this, p.cast<int>()});
                    ++trackNum;
                }
            } else {
                ++depthOutofRangeNum;
            }
        }
    }
    cout << "epipolarSearchNoneNum, depthOutofRangeNum, trackNum: "
         << epipolarSearchNoneNum << " " << depthOutofRangeNum << " "
         << trackNum << endl;
    return trackNum;
}

int KeyFrame::ReuseLandmark(KeyFrame* kf1) {
    // 给新的KF2预分配内存
    landmark_ = vector<Landmark*>(unPx_[0].size(), nullptr);
    const int w = grayImg_.cols, h = grayImg_.rows;

    auto Project2Edge = [this, &w](const Eigen::Vector2i px) -> bool {
        // 允许的像素偏差

        const vector<Eigen::Vector2i> xy = {{0, 1},   {0, -1}, {-1, 0},
                                            {1, 0},   {-1, 1}, {1, 1},
                                            {-1, -1}, {1, -1}};
        for (const Eigen::Vector2i& p : xy) {
            Eigen::Vector2i px2 = px + p;
#if USE_POINT_MAP_ID
            if (pointMapId_.count(px2)) {
#else
            if (pointMapId_.count(px2.y() * w + px2.x())) {
#endif
                return true;
            }
        }
        return false;
    };

    const int debugBin = 5;
    vector<vector<Eigen::Vector2i> > debugProj1(5), debugProj2(5);
    const int debugPart = edgeImg_[0].cols / 5;  // 显示分区

    int reuseLandmarkNum = 0;
    const vector<Landmark*>& landmark = kf1->landmark_;
    for (int i = 0; i < landmark.size(); ++i) {
        if (landmark[i] == nullptr || landmark[i]->IsOutOfRange()) {
            // 未收敛的landmark也当作可追踪的，因其在未来可能收敛
            continue;
        }
        Landmark* pc1 = landmark[i];

        // 更新的是host帧下的depth
        const Eigen::Vector3d pc2 = Tcw_ * pc1->GetPw();
        if (pc2.z() < config->minDepth || pc2.z() > config->maxDepth) {
            continue;
        }

        const Eigen::Vector2i px2 = cam_->Project2PixelPlane(pc2).cast<int>();
        const int x = px2.x(), y = px2.y();
#if USE_POINT_MAP_ID
        if (pointMapId_.count(px2) || Project2Edge(px2)) {
            const int vecId = pointMapId_[px2];
#else
        const int id = y * w + x;
        if (pointMapId_.count(id) || Project2Edge(px2)) {
            const int vecId = pointMapId_.at(id);
#endif
            if (landmark_[vecId] != nullptr) {
                // TODO: 选一个更好的，或者按照先来后到
                continue;
            }
            const uint64_t d2 = descriptor_[vecId];
            const uint64_t d1 = pc1->descriptor_;
            uint64_t score = CalculateDescriptorScore(d1, d2);
            if (score < config->goodDescriptorDist) {
                // 增加相互观测
                landmark_[vecId] = pc1;
                pc1->target_.insert({this, px2});

                const int debugId = pc1->uv_.x() / debugPart;
                debugProj1[debugId].push_back(pc1->uv_);
                debugProj2[debugId].push_back(unPx_[0][vecId]);

                ++reuseLandmarkNum;
            }
        }
    }
    for (int i = 0; i < debugBin; ++i) {
        const string name("reuse landmark match part");  // +to_string(i));
        cv::namedWindow(name);
        // DrawMatch(kf1->edgeImg_[0], edgeImg_[0], debugProj1[i], debugProj2[i],
        //     name, 1, 1);
    }
    cv::destroyAllWindows();
    return reuseLandmarkNum;
}

void KeyFrame::Update(const Eigen::Vector3d& delta_q,
                      const Eigen::Vector3d& delta_t) {
    Twc_.Update(delta_q, delta_t);
    Tcw_ = Twc_.Inverse();
}

void KeyFrame::SetTwc(const Pose& Twc) {
    Twc_ = Twc;
    Tcw_ = Twc.Inverse();
}

double KeyFrame::CullingBadDepth(KeyFrame* kf2) {
    // 对已经收敛的深度点做检验，若连续3次重投影校验失败，则删除！
    if (config->useDepthImage) {
        return 0.;
    }
    int convergeNum = 0, badNum = 0;
    const Pose T21 = kf2->Twc_.Inverse() * Twc_;
    for (int i = 0; i < static_cast<int>(landmark_.size()); ++i) {
        Landmark* lk1 = landmark_[i];
        if (lk1 == nullptr || lk1->IsOutOfRange() || !lk1->Converge() ||
            lk1->lastFrameMatchPx_.isApproxToConstant(0) ||
            lk1->passReprojectCheck_) {
            continue;
        }

        const Eigen::Vector3d pc1 = lk1->GetPc();
        const Eigen::Vector3d pc2 = T21 * pc1;
        const Eigen::Vector2d px2 = cam_->Project2PixelPlane(pc2);
        const double residual = (px2 - lk1->lastFrameMatchPx_).head(2).norm();
        if (residual > 1.0) {
            lk1->continousFailCheckNum_++;
        } else {
            lk1->continousPassCheckNum_++;
        }
        if (lk1->CheckInvDepthQualitySuccessByProject()) {
            ++convergeNum;
        } else {
            ++badNum;
        }
        /*
        const double depthRatio = pc2.z() / pc1.z();
        if (depthRatio < config->minDepthCompareRatio ||
            depthRatio > config->maxDepthCompareRatio) {
            // 距离变化过大，导致深度差异大，不能再比较了
            continue;
        }

        const Eigen::Vector2i px2 = cam_->Project2PixelPlane(pc2).cast<int>();
        if (!InRange(kf2->grayImg_, px2)) {
            // 投影点不在视野内是正常的
            continue;
        }

        bool isBad = false;
        const double residual = CalculatePatchSSD(this, kf2, lk1->uv_, px2);
        // if(residual > config->maxDescriptorDist) {
        //     lk1->depthCov_ *= 1.1;
        //     isBad = true;
        // }

        const double dist = kf2->dist_[0].at<float>(px2.y(), px2.x());
        if (dist > 5) {
            lk1->SetOutOfRange();
        }
        if (dist > config->maxTrackProjectPixelError && residual > 40) {
            // if(dist > config->maxTrackProjectPixelError) {
            // lk1->SetOutOfRange();
            // 投影点误差大的就给它重置
            // lk1->depthCov_ = pow(config->maxDepth, 2);
            ++lk1->checkTime_;
            lk1->invDepthCov_ *=
                1.2;  // sqrt(dist)*sqrt(dist) pow(1.1, int(dist));
            isBad = true;
        }

        if (isBad) {
            ++badNum;
            lk1->UpdateUncertainty(false);
        }

#ifdef USE_SSD
        // 考虑到图像远近，似乎不能使用这个条件？
        // if( CalculatePatchSSD(this, kf2, lk1->uv_, px2) > config->maxSSDdist) {
        //     lk1->SetOutOfRange();
        //     ++badNum;
        //     continue;
        // }
#endif
*/
    }
    return double(badNum) / convergeNum;
}

double KeyFrame::FindMatchesWithEpipolarConstraintOnImagePlane(
    const KeyFrame* kf2, Landmark* lk1, Eigen::Vector2d& bestPx2,
    Eigen::Vector2d& farPx2, Eigen::Vector2d& nearPx2,
    Eigen::Vector2d& epipolarP1, Eigen::Vector2d& ep2) {
    if (lk1 == nullptr || lk1->IsOutOfRange()) {
        // cout << "Error lk1 is nullptr or out of range!\n";
        AddReportElement("lk1 is nullptr or out of range!");
        return EpipolarMatchType::nanValueNOstereoVisionIssue;
    }

    const Pose T21 = kf2->Twc_.Inverse() * Twc_;
    const Pose T12 = T21.Inverse();
    const double fx = cam_->fx_, fy = cam_->fy_, cx = cam_->cx_, cy = cam_->cy_;

    // 求KF1像素平面上对应极线，直接把O2当作O1相机系下的一点，那么极点位置不就轻松求出来了
    // 极线就是极点与p1的交点
    // 已知 t12, 那么KF2光心与KF1归一化平面的交点可求，但是当z[2] = 0时：就让极点都乘以t12嘛
    const Eigen::Vector3d& t12 = T12.t_wb_;
    Eigen::Vector2d p1 = lk1->uv_.cast<double>();
    // ep1 = (x, y)*t12.z - 极点e1*t12.z
    // 注意：这个极线是像素平面上放大 t12.z 倍后的方向向量
    Eigen::Vector2d ep1{-fx * t12[0] + t12[2] * (p1.x() - cx),
                        -fy * t12[1] + t12[2] * (p1.y() - cy)};
    if (abs(t12.z()) > 1e-9) {
        ep1 /= t12.z();
        if (ep1.norm() < 1.0) {
            // cout << "Error ep1 norm: " << ep1.norm() << " < 1 pixel!\n";
            AddReportElement("ep1 norm < 1!");
            return EpipolarMatchType::nanValueNOstereoVisionIssue;
        }
        epipolarP1 = p1 - ep1;
    } else {
        // 此时极线是无限长的，不需要判断长度
        return EpipolarMatchType::outOFboundaryORabnormalDepth;
    }
    ep1.normalize();

    const Eigen::Vector3d priorPc2 = T21 * lk1->GetPc();
    const double depthScale = priorPc2.z() * lk1->invZ_;
    if (!(depthScale > 0.7f && depthScale < 1.4f) && 0) {
        // cout << "Error depthScale: " << depthScale << "\n";
        AddReportElement("depthScale error!");
        return EpipolarMatchType::outOFboundaryORabnormalDepth;
    }
    // 根据近大远小的规则调整窗口范围
    if (depthScale > 1.0) {
        ep1 *= depthScale;
    }

    // 虽然正深度分布不能直接转换，但是深度范围是可以这么确定的
    const double stddev = sqrt(lk1->invDepthCov_);
    double maxZ1 = min(100.0, GetPositiveDepth(lk1->invZ_ - 3.0 * stddev));
    double minZ1 = max(0.01, GetPositiveDepth(lk1->invZ_ + 3.0 * stddev));
    const Eigen::Vector3d farPc1 = cam_->InverseProject(lk1->uv_, maxZ1);
    const Eigen::Vector3d nearPc1 = cam_->InverseProject(lk1->uv_, minZ1);
    const Eigen::Vector3d farPc2 = T21 * farPc1;
    const Eigen::Vector3d nearPc2 = T21 * nearPc1;
    if (farPc2.z() < nearPc2.z()) {
        // cout << "Error far z: " << farPc2.z() << " < near z: " << nearPc2.z() << "\n";
        AddReportElement("Far z < near z");
        return EpipolarMatchType::outOFboundaryORabnormalDepth;
    }

    farPx2 = cam_->Project2PixelPlane(farPc2);
    nearPx2 = cam_->Project2PixelPlane(nearPc2);
    // 从 far->near 的方向向量，ep1极线向量的方向相对图像需要与此保持一致
    // 极点对应着最小深度投影(因为是归一化平面交点)
    ep2 = nearPx2 - farPx2;
    const double ep2Len = ep2.norm();
    ep2 = ep2.normalized();
    if (depthScale < 1.0) {
        ep2 /= depthScale;
    }

    const int desLen = config->descriptorPatchLen;
    const int midLen = desLen / 2;
    // 检验一下描述子是否在范围内
    Eigen::Vector2d p1Start = p1 - midLen * ep1, p1End = p1 + midLen * ep1;
    if (!InRange(grayImg_, p1Start.cast<int>())) {
        // cout << "Error p1Start not in image!\n";
        AddReportElement("Error p1Start not in image!");
    }
    if (!InRange(grayImg_, p1End.cast<int>())) {
        // cout << "Error p1End!\n";
        AddReportElement("Error p1End!");
        return EpipolarMatchType::outOFboundaryORabnormalDepth;
    }

    // 保证双目图像上极线方向相对于图像是从左到右还是从右到左保持一致
    CheckEpipolarLineDirection(ep2, ep1);

    // OK，接下来在对极线上等距取5个点，据此来计算SSD
    ep1 *= config->minSearchStep;
    vector<double> v1 = CalculateDescriptor(grayImg_, p1, ep1, desLen);

    // OK， 我们需要限制一下KF2上的极线范围，这是最重要的先验，极线搜索距离越短，受相对旋转的影响越小
    const double maxEpipolarLen = config->maxEpipolarSearchLine,
                 minEpipolarLen = config->minEpipolarSearchLine;
    const double cutLen = ep2Len - maxEpipolarLen;
    const double expandLen = minEpipolarLen - ep2Len;
    if (cutLen > 0) {
        // 只能修改最近点
        // nearPx2 -= cutLen * ep2;
        // nearPx2 = farPx2 - cutLen * ep2;
        nearPx2 = farPx2 + maxEpipolarLen * ep2;
    } else if (expandLen > 0) {
        const double halfLen = 0.5 * expandLen;
        nearPx2 += halfLen * ep2;
        farPx2 -= halfLen * ep2;
    }

    // 最远点若不在投影范围内，那么意味着视野范围受限？直接返回
    const Mat& img2 = kf2->grayImg_;
    if (!InRange(img2, farPx2.cast<int>())) {
        // cout << "Error farPx2 not in image!\n";
        AddReportElement("Error farPx2 not in image!");
        return EpipolarMatchType::outOFboundaryORabnormalDepth;
    }
    if (!InRange(img2, nearPx2.cast<int>())) {
        // 将最近点移动到图像内
        if (!MoveNearPx2IntoBoundary(nearPx2, ep2, farPx2)) {
            // cout << "Error nearPx2 not in image!\n";
            AddReportElement("Error nearPx2 not in image!");
            return EpipolarMatchType::outOFboundaryORabnormalDepth;
        }
    }

    ep2 *= config->minSearchStep;
    return SearchBestMatchAlongEpipolarLine(farPx2, nearPx2, ep2, *kf2, v1,
                                            bestPx2);
}

double KeyFrame::SearchBestMatchAlongEpipolarLine(
    const Eigen::Vector2d& farPx2, const Eigen::Vector2d& nearPx2,
    const Eigen::Vector2d& ep2, const KeyFrame& kf2, const vector<double>& v1,
    Eigen::Vector2d& bestPx2) {
    double bestScore = 1e9, secondBestScore = 1e9;
    Eigen::Vector2d bestP2{1000, 1000}, secondBestP2{1000, 1000};

    Eigen::Vector2d p2 = farPx2;
    const int desLen = config->descriptorPatchLen;
    const int midLen = desLen / 2;
    Eigen::Vector2d p2End = p2 - midLen * ep2, p2Start = p2 + midLen * ep2;
    // 保证端点在图像范围内
    const cv::Mat& grayImg2 = kf2.grayImg_;
    if (!InRange(grayImg2, p2End.cast<int>())) {
        p2End = p2;
        p2 = p2Start;
        p2Start = p2 + midLen * ep2;
    }
    vector<double> v2 = CalculateDescriptor(grayImg2, p2, ep2, desLen);

    double s2 = 0, avg2 = 0;  // 可以使用滑窗计算
    for (int i = 0; i < desLen; ++i) {
        s2 += v2[i];
    }
    avg2 = s2 / desLen;

    if (ep2.norm() < 0.99) {
        cout << "Error ep2 norm: " << ep2.norm() << " < 0.99!\n";
        AddReportElement("ep2 norm < 1!");
        return EpipolarMatchType::outOFboundaryORabnormalDepth;
    }

    // 通过判断incx，incy的正负，来判断循环终止条件
    const double avg1 = std::accumulate(v1.begin(), v1.end(), 0.) / v1.size();
    while (1) {
        if (InRange(grayImg2, p2.cast<int>()) &&
            kf2.dist_[0].at<float>(int(p2.y()), int(p2.x())) < 1.0) {
            // 已经保证端点在边界范围内，这里无需再判断
            const double score = CalculateSSD(v1, v2, avg1, avg2, desLen);

            if (score < bestScore) {
                secondBestScore = bestScore;
                secondBestP2 = bestP2;
                bestScore = score;
                bestP2 = p2;
            } else if (score < secondBestScore) {
                secondBestScore = score;
                secondBestP2 = p2;
            }
        }

        // 根据滑窗的思路，我们可以只移动一个即可
        p2End += ep2;    // 判断边界
        p2 += ep2;       // 移动关键点
        p2Start += ep2;  // 判断边界及移动滑窗
        s2 -= v2[0];     // v2[0] 对应的可是 p2End 点
        for (int i = 1; i < static_cast<int>(v2.size()); ++i) {
            v2[i - 1] = v2[i];
        }
        const bool inSearchRange = (ep2[0] > 0) == (p2Start[0] < nearPx2[0]) &&
                                   (ep2[1] > 0) == (p2Start[1] < nearPx2[1]);
        if (!inSearchRange) {
            break;
        }
        const double v2Back = BilinearInterpolate<uchar>(grayImg2, p2Start);
        v2.back() = v2Back;
        s2 += v2.back();
        avg2 = s2 / desLen;
    }

    const bool smallScore = bestScore < config->maxDescriptorDist * desLen;
    if (!smallScore) {
        // cout << "Error smallScore: " << smallScore << "\n";
        AddReportElement("Error smallScore!");
        return EpipolarMatchType::occulsionORnoBestMatch;
    }

    const bool goodScore =
        bestScore < config->best2SecondRatio * secondBestScore;
    if (!goodScore) {
        // cout << "Error goodScore\n";
        AddReportElement("Error goodScore!");
        return EpipolarMatchType::repeatTextureORbadDepth;
    }

    double badDist = false;
    if (secondBestScore < 1e8) {
        badDist = (bestP2 - secondBestP2).norm() > config->best2SecondDist;
    }
    if (badDist) {
        // cout << "Error badDist: " << badDist << "\n";
        AddReportElement("Error badDist!");
        return EpipolarMatchType::repeatTextureORbadDepth;
    }

    if (goodScore && !badDist) {
        bestPx2 = bestP2;
        AddReportElement("Find Match succeed!");
        return bestScore;
    }

    AddReportElement("Final Fail!");
    return EpipolarMatchType::outOFboundaryORabnormalDepth;
}

bool KeyFrame::MoveNearPx2IntoBoundary(Eigen::Vector2d& pClose,
                                       const Eigen::Vector2d& ep2,
                                       const Eigen::Vector2d& pFar) {
#define SAMPLE_POINT_TO_BORDER 7
    const int width = edgeImg_[0].cols, height = edgeImg_[0].rows;
    const double incx = ep2[0], incy = ep2[1];
    if (pClose[0] <= SAMPLE_POINT_TO_BORDER ||
        pClose[0] >= width - SAMPLE_POINT_TO_BORDER ||
        pClose[1] <= SAMPLE_POINT_TO_BORDER ||
        pClose[1] >= height - SAMPLE_POINT_TO_BORDER) {
        if (pClose[0] <= SAMPLE_POINT_TO_BORDER) {
            float toAdd = (SAMPLE_POINT_TO_BORDER - pClose[0]) / incx;
            pClose[0] += toAdd * incx;
            pClose[1] += toAdd * incy;
        } else if (pClose[0] >= width - SAMPLE_POINT_TO_BORDER) {
            float toAdd = (width - SAMPLE_POINT_TO_BORDER - pClose[0]) / incx;
            pClose[0] += toAdd * incx;
            pClose[1] += toAdd * incy;
        }

        if (pClose[1] <= SAMPLE_POINT_TO_BORDER) {
            float toAdd = (SAMPLE_POINT_TO_BORDER - pClose[1]) / incy;
            pClose[0] += toAdd * incx;
            pClose[1] += toAdd * incy;
        } else if (pClose[1] >= height - SAMPLE_POINT_TO_BORDER) {
            float toAdd = (height - SAMPLE_POINT_TO_BORDER - pClose[1]) / incy;
            pClose[0] += toAdd * incx;
            pClose[1] += toAdd * incy;
        }

        // get new epl length
        // OK, 根据极线端点我们就能计算极线长度了
        float fincx = pClose[0] - pFar[0];
        float fincy = pClose[1] - pFar[1];
        float newEplLength = sqrt(fincx * fincx + fincy * fincy);

        // test again
        // 如果curF上对应的极线太短，意味着旋转太大？导致无法找到正确匹配？
        // oob: out of border, 超出范围？
        if (pClose[0] <= SAMPLE_POINT_TO_BORDER ||
            pClose[0] >= width - SAMPLE_POINT_TO_BORDER ||
            pClose[1] <= SAMPLE_POINT_TO_BORDER ||
            pClose[1] >= height - SAMPLE_POINT_TO_BORDER ||
            newEplLength < 8.0f) {
            return false;
        }
    }
    return true;
}

void KeyFrame::ReleaseMat() {
    grayImg_.release();
    for (int i = 0; i < edgeImg_.size(); ++i) {
        edgeImg_[i].release();
        dist_[i].release();
        dx_[i].release();
        dy_[i].release();
    }
}

void KeyFrame::FuseDepth() {

    if (config->useDepthImage) {
        return;
    }
    const int fusePatchLen = 5;
    const int midLen = fusePatchLen / 2;

    vector<Landmark> readLk(landmark_.size());
    for (const Landmark* lk : landmark_) {
        readLk.emplace_back(*lk);
    }
#if USE_POINT_MAP_ID
    const unordered_map<Eigen::Vector2i, int, TupleHash>& readPointMapId =
        pointMapId_;
#else
    const unordered_map<int, int>& readPointMapId = pointMapId_;
#endif

    constexpr bool removeOcclusion = true;
    const int w = grayImg_.cols;

    for (int i = 0; i < static_cast<int>(landmark_.size()); ++i) {
        Landmark* lk1 = landmark_[i];
        const Eigen::Vector2i px1 = lk1->uv_;
        const double& iDepth1 = lk1->invZ_;
        const double stddev1 = sqrt(lk1->invDepthCov_);

        int occlusionCount = 0;
        for (int x = -midLen; x <= midLen; ++x) {
            for (int y = -midLen; y <= midLen; ++y) {
                const Eigen::Vector2i px2 = px1 + Eigen::Vector2i(x, y);

#if USE_POINT_MAP_ID
                if (!readPointMapId.count(px2)) {
                    continue;
                }
                const Landmark& lk2 = readLk[readPointMapId.at(px2)];
#else
                const int id = px2.y() * w + px2.x();
                if (!readPointMapId.count(id)) {
                    continue;
                }
                const Landmark& lk2 = readLk[readPointMapId.at(id)];
#endif
                if (!lk2.Converge() || lk2.IsOutOfRange()) {
                    continue;
                }

                const double& iDepth2 = lk2.invZ_;
                if (iDepth2 > iDepth1 + stddev1 * 2.0) {
                    ++occlusionCount;
                }
                if (occlusionCount > int(fusePatchLen * fusePatchLen * 0.6) &&
                    removeOcclusion) {
                    lk1->SetOutOfRange();
                    continue;
                }

                if (!lk1->FuseInvDepth(lk2)) {
                }
            }
        }
    }
}
