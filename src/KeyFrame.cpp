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

#if defined(WRITE_MATCH_PAIR_IMAGE)
cv::VideoWriter KeyFrame::debugVideoWriter;
constexpr int kDrawMatchNumEachFrame = 100;
cv::VideoWriter KeyFrame::debugTriangulateWriter;
#endif
cv::Mat KeyFrame::map1;
cv::Mat KeyFrame::map2;
Pose KeyFrame::Tc0w;

class Landmark;

KeyFrame::KeyFrame(const cv::Mat& img, const Pose& Twc,
                   std::shared_ptr<Camera> cam, const int id, const int level)
    : id_(id),
      grayImg_(img),
      cam_(cam),
      Twc_{Twc},
      Tcw_(Twc.Inverse()),
      priorTwc_(Twc),
      level_(level) {
    GenerateUndistordMap();
    cv::remap(grayImg_, grayImg_, map1, map2, cv::INTER_LINEAR);
    debugGrayImg_ = grayImg_.clone();
}

KeyFrame::KeyFrame(const KeyFrame& f)
    : id_(f.id_),
      grayImg_(f.grayImg_),
      cam_(f.cam_),
      Twc_(f.Twc_),
      Tcw_(f.Tcw_),
      priorTwc_(f.priorTwc_),
      level_(f.level_),
      unKeypoints_(f.unKeypoints_),
      outOfRange_(f.outOfRange_),
      convergeEdgeNum_(f.convergeEdgeNum_) {
    // vector内的堆内存需要先释放
    // 不能这样子，这是构造函数，默认的内存应该是干净的，
    // 否则你应该调用赋值构造

    // 照理说这里不应该执行
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

void KeyFrame::operator=(const KeyFrame& f) {
#if 1
    id_ = f.id_;
    grayImg_ = f.grayImg_;
    cam_ = f.cam_;
    Twc_ = f.Twc_;
    Tcw_ = f.Tcw_;
    priorTwc_ = f.priorTwc_;
    level_ = f.level_;
    unKeypoints_ = f.unKeypoints_;
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
    //optFlw_ = f.optFlw_; // 不能直接这么设
    SetOpticalFlowStructCurFrame();
#else
    ReleaseMat();
    // 这样会导致cv::Mat等堆内存无法释放
    new (this) KeyFrame(f);
#endif
    depthImage_ = f.depthImage_;
    debugGrayImg_ = f.debugGrayImg_;
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

void KeyFrame::SetOpticalFlowStructCurFrame() {
    optFlw_.prevImg_ = grayImg_;
    optFlw_.prevPts_.reserve(landmark_.size());
    optFlw_.trackLandmark_.reserve(landmark_.size());
    for (const auto& p : landmark_) {
        // 添加landmark对跟踪成功点的相互观测
        optFlw_.prevPts_.emplace_back(cv::Point2f(p->uv_.x(), p->uv_.y()));
        optFlw_.trackLandmark_.push_back(p);
    }
}

void KeyFrame::GenerateUndistordMap() {
    if (map1.empty()) {
        cv::Mat D;
        if (config->model == "fisheye") {
            D = (cv::Mat_<float>(4, 1) << cam_->k1_, cam_->k2_, cam_->k3_,
                 cam_->k4_);
        } else if (config->model == "pinhole") {
            D = (cv::Mat_<float>(5, 1) << cam_->k1_, cam_->k2_, cam_->k3_,
                 cam_->k4_, cam_->k5_);
        }
        cv::Mat R = cv::Mat::eye(3, 3, CV_32F);
        // 去畸变后，可以使用原有的内参，也可以使用自己定义的新内参，rebvo就是自己定义了新的 zf=(fx+fy)*0.5
        cv::Mat K = (cv::Mat_<float>(3, 3) << cam_->fx_, 0, cam_->cx_, 0,
                     cam_->fy_, cam_->cy_, 0, 0, 1);

        // 使用新的内参投影，避免出现黑色的部分
        double newFx = (cam_->fx_ + cam_->fy_) * 0.5 * 1.1;
        cv::Mat newK = (cv::Mat_<float>(3, 3) << newFx, 0, cam_->cx_, 0, newFx,
                        cam_->cy_, 0, 0, 1);

        // 畸变模板应该只需要计算一次！！！
        cv::initUndistortRectifyMap(K, D, cv::Mat(), newK,
                                    cv::Size(grayImg_.cols, grayImg_.rows),
                                    CV_8UC1, map1, map2);

        cam_->UpdateIntrinsicParam(newFx, newFx);
    }
}

void KeyFrame::ExtractFastPoints(const OpticalFlowStruct& lastKFoptFlw) {
    cv::Mat search;
    if (!lastKFoptFlw.prevImg_.empty()) {
        search = cv::Mat::zeros(lastKFoptFlw.prevImg_.size(), CV_8UC1);
        // 当前关键帧追踪到当前帧的特征点，不要重复创建
        // 遍历当前帧被跟踪到的特征点
        for (const cv::Point2f& p : lastKFoptFlw.prevPts_) {
            search.ptr<uchar>(int(p.y + 0.5))[int(p.x + 0.5)] = 255;
        }

        for (const cv::Point2f& p : lastKFoptFlw.prevHistoryPts_) {
            search.ptr<uchar>(int(p.y + 0.5))[int(p.x + 0.5)] = 255;
        }

        //cv::imshow("search", search);
        //cv::waitKey();
    }

    auto CanGenerateKeyPoint = [&search](const cv::Point2f& p) -> bool {
        int x = int(p.x + 0.5);
        int y = int(p.y + 0.5);
        for (int i = -1; i < 2; ++i) {
            for (int j = -1; j < 2; ++j) {
                int xi = x + i;
                int yi = y + j;
                if (xi < 1 || yi < 1 || xi >= search.cols ||
                    yi >= search.rows) {
                    return false;
                }
                if (search.ptr<uchar>(yi)[xi] != 0) {
                    return false;
                }
            }
        }

        return true;
    };

    // 创建FAST检测器
    cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create(
        config->fastTh, true, cv::FastFeatureDetector::TYPE_9_16);
    vector<cv::KeyPoint> pts;
    detector->detect(grayImg_, pts);
    pts.reserve(5000);
    unKeypoints_.reserve(pts.size());
    for (const cv::KeyPoint& p : pts) {
        // 注意：需要把上一KF的optFlw一直保留而不能重置
        if (search.empty() || CanGenerateKeyPoint(p.pt)) {
            cv::circle(debugGrayImg_, p.pt, 2, kColor.at("white"));
            unKeypoints_.emplace_back(p.pt.x, p.pt.y);
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

void KeyFrame::OpticalFlowTrackExcute(const cv::Mat& prevImg,
                                      const cv::Mat& curImg,
                                      vector<cv::Point2f>& prevPts,
                                      vector<Landmark*>& prevTrackLandmark) {
    vector<cv::Point2f> nextPts;
    vector<uchar> status;
    vector<float> error;

    cv::calcOpticalFlowPyrLK(prevImg, curImg, prevPts, nextPts, status, error);
    vector<Landmark*> trackLandmark;
    const auto debugPts1 = prevPts;
    prevPts.clear();
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i] == 1 && error[i] < 10) {
            // 重新赋值landmark在当前帧上的观测
            prevPts.emplace_back(nextPts[i]);
            trackLandmark.emplace_back(prevTrackLandmark[i]);
            Eigen::Vector2i p1(int(debugPts1[i].x), int(debugPts1[i].y));
            Eigen::Vector2i p2(int(nextPts[i].x), int(nextPts[i].y));
            DrawBestMatchEachFrame(p1, p2, curImg, true);
        }
    }
    prevTrackLandmark = trackLandmark;
}

void KeyFrame::OpticalFlowTrackLandmark(const KeyFrame& f2) {

    if (optFlw_.prevPts_.empty()) {
        cout << "here optFlw_.prevPts_ should not be empty!!!";
        exit(-1);
    }

    // 上一关键帧对当前帧的跟踪结果
    OpticalFlowTrackExcute(optFlw_.prevImg_, f2.grayImg_, optFlw_.prevPts_,
                           optFlw_.trackLandmark_);
    WriteDebugImage2VideoEachFrame(f2.id_, "lastKF_track_result.avi");

    // 历史关键帧对当前帧的跟踪结果
    cout << "optFlw_.prevHistoryPts_.size: " << optFlw_.prevHistoryPts_.size()
         << endl;
    if (!optFlw_.prevHistoryPts_.empty()) {
        OpticalFlowTrackExcute(optFlw_.prevImg_, f2.grayImg_,
                               optFlw_.prevHistoryPts_,
                               optFlw_.trackHistoryLandmark_);
        WriteDebugImage2VideoEachFrame(f2.id_, "historyKF_track_result.avi");
    }

    optFlw_.prevImg_ = f2.grayImg_;
    // 跟踪成功后，重新赋值
    cout << "optical flow tracked point num lastKf: "
         << optFlw_.trackLandmark_.size() << endl;
    cout << "optical flow tracked point num history Kf: "
         << optFlw_.trackHistoryLandmark_.size() << endl;
}

double KeyFrame::TrackWithOpticalFlow(const KeyFrame& kf2, int& findMatchNum) {
    OpticalFlowTrackLandmark(kf2);
    findMatchNum = optFlw_.GetTrackFeatureNum();
    return optFlw_.GetTrackFeatureRatio();
}

void KeyFrame::CopyStatus() {
    TcwBack_ = Tcw_;
    TwcBack_ = Twc_;
}

void KeyFrame::BackUpStatus() {
    Tcw_ = TcwBack_;
    Twc_ = TwcBack_;
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
                                      const cv::Mat& debugImg2,
                                      const bool drawOpticalFlow) {
    if (videoBestMatchDebugImg_.empty()) {
        videoBestMatchDebugImg_ =
            cv::Mat(debugGrayImg_.rows, debugGrayImg_.cols * 2, CV_8UC3,
                    cv::Scalar{0, 0, 0});
        cv::Mat im1, im2;
        if (drawOpticalFlow) {
            cvtColor(optFlw_.prevImg_, im1, cv::COLOR_GRAY2BGR);
        } else {
            cvtColor(debugGrayImg_, im1, cv::COLOR_GRAY2BGR);
        }
        cvtColor(debugImg2, im2, cv::COLOR_GRAY2BGR);
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

void KeyFrame::WriteDebugImage2VideoEachFrame(const int kf2Id,
                                              const string& debugVideoName) {
    if (videoBestMatchDebugImg_.empty()) {
        return;
    }
    int start_text_row = 20;
    int step_text_row = 10;
    cv::putText(videoBestMatchDebugImg_,
                fmt::format("kf id: {} match point curF id: {}_{}", id_, kf2Id,
                            debugVideoName),
                cv::Point(10, (start_text_row += step_text_row)),
                cv::FONT_ITALIC, 1.0, kColor.at("red"), 1);
    if (!videoEpipolarMatchDebugImg_.empty()) {
        cv::putText(videoEpipolarMatchDebugImg_,
                    "match ep curF id: " + to_string(kf2Id),
                    cv::Point(10, start_text_row), cv::FONT_ITALIC, 1.0,
                    kColor.at("red"), 1);
    }

    if (!videoEpipolarFailMatchDebugImg_.empty()) {
        cv::putText(videoEpipolarFailMatchDebugImg_,
                    "fail ep curF id: " + to_string(kf2Id),
                    cv::Point(10, start_text_row), cv::FONT_ITALIC, 1.0,
                    kColor.at("red"), 1);
    }

    // 初始化边缘匹配的debug视频写入器
    if (!KeyFrame::debugVideoWriter.isOpened()) {
        string videoPath = config->debugMessageSaveFolder;
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

    //debugVideoWriter.write(videoEpipolarMatchDebugImg_);
    debugVideoWriter.write(videoBestMatchDebugImg_);
    //debugVideoWriter.write(videoEpipolarFailMatchDebugImg_);
    //videoEpipolarMatchDebugImg_.release();
    videoBestMatchDebugImg_.release();
    //videoEpipolarFailMatchDebugImg_.release();
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
        cv::Mat im1, im2;
        cvtColor(debugGrayImg_, im1, cv::COLOR_GRAY2BGR);
        cvtColor(debugImg2, im2, cv::COLOR_GRAY2BGR);
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
        cv::Mat im1, im2;
        cvtColor(debugGrayImg_, im1, cv::COLOR_GRAY2BGR);
        cvtColor(debugImg2, im2, cv::COLOR_GRAY2BGR);
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
    cv::Mat im1, im2;
    cvtColor(debugGrayImg_, im1, cv::COLOR_GRAY2BGR);
    cvtColor(debugImg2, im2, cv::COLOR_GRAY2BGR);
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
    const Eigen::Vector2i& kp1 = lk1.uv_.cast<int>();
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

size_t KeyFrame::InitializeLandmark(const KeyFrame* lastKf) {
    // TODO：需要考虑由三角化前、后帧生成地图点而创建的landmark？
    if (landmark_.empty()) {
        const OpticalFlowStruct& lastKFoptFlw =
            lastKf == nullptr ? OpticalFlowStruct() : lastKf->optFlw_;
        ExtractFastPoints(lastKFoptFlw);
        // initialKF会有该种情况
        landmark_.resize(unKeypoints_.size(), nullptr);
    }

    for (size_t i = 0; i < unKeypoints_.size(); ++i) {
        if (landmark_[i] != nullptr && !config->useDepthImage &&
            !config->debugWithTrueDepthImage) {
            continue;
        }
        const int x = int(unKeypoints_[i].x() + 0.5);
        const int y = int(unKeypoints_[i].y() + 0.5);
        // shared_ptr<KeyFrame>(this)会导致多源智能指针，它会释放多次KeyFrame导致报错
        // 若需要使用智能指针，必须保证this在此前已经由一个智能指针管理，然后使用shared_from_this()来获取，否则只能使用原始指针
        // landmark_.push_back(make_shared<Landmark>(upx, make_shared<KeyFrame>(this), cam_, 1.0) ); [ERROR double free]

        const uint64_t descriptor = 0;
        if (!config->useDepthImage && !config->debugWithTrueDepthImage &&
            depthImage_.empty()) {
            landmark_[i] = new Landmark(unKeypoints_[i], this, cam_, descriptor,
                                        kInitInvDepth);
        } else {
            // double d = static_cast<double>(depthImage_.ptr<ushort>(y)[x]);
            double d = static_cast<double>(depthImage_.at<ushort>(y, x));
            if (d == 0) {
                landmark_[i] = new Landmark(unKeypoints_[i], this, cam_,
                                            descriptor, kInitInvDepth);

            } else {
                d /= config->depthFactor;
                landmark_[i] = new Landmark(unKeypoints_[i], this, cam_,
                                            descriptor, kInitInvDepth);
                landmark_[i]->trueDepth_ = d;
                if (config->useDepthImage) {
                    landmark_[i]->invDepthCov_ = 0.005;
                    landmark_[i]->obvTime_ = 1e3;
                    landmark_[i]->invZ_ = 1.0 / d;
                }
            }
            // host帧也要增加与landmark的相互观测
            landmark_[i]->target_.insert({this, unKeypoints_[i]});
        }
    }

    // 初始化当前新建关键帧进行光流跟踪所需的结构，仅针对当前KF
    SetOpticalFlowStructCurFrame();  // 设置光流跟踪的匹配结构

    //此外还需把上一关键帧中保留的光流及历史关键帧的光流跟踪结果合并到当前关键帧
    if (lastKf != nullptr) {
        // 添加上一关键帧新提取的关键点
        for (size_t i = 0; i < lastKf->optFlw_.prevPts_.size(); ++i) {

            optFlw_.prevHistoryPts_.emplace_back(lastKf->optFlw_.prevPts_[i]);
            optFlw_.trackHistoryLandmark_.emplace_back(
                lastKf->optFlw_.trackLandmark_[i]);
        }

        for (size_t i = 0; i < lastKf->optFlw_.prevHistoryPts_.size(); ++i) {
            // 添加历史关键帧跟踪上的关键点
            optFlw_.prevHistoryPts_.emplace_back(
                lastKf->optFlw_.prevHistoryPts_[i]);
            optFlw_.trackHistoryLandmark_.emplace_back(
                lastKf->optFlw_.trackHistoryLandmark_[i]);
        }
    }

    optFlw_.SetTotalFeatureCreated();

    return optFlw_
        .GetTrackFeatureNum();  // 历史关键帧和当前关键帧在当前灰度图上提取到的关键点数量
}

void KeyFrame::Update(const Eigen::Vector3d& delta_q,
                      const Eigen::Vector3d& delta_t) {
    Twc_.Update(delta_q, delta_t);
    Tcw_ = Twc_.Inverse();
}

void KeyFrame::SetTwc(const Pose& Twc, const bool printDiff) {
    if(printDiff && Tc0w.t_wb_.isApproxToConstant(0)) {
        // 设置运行时世界系到数据集世界系的变换
        Tc0w = priorTwc_.Inverse();
    }

    Twc_ = Twc;
    Tcw_ = Twc.Inverse();
    if (printDiff) {
        const Pose diff = (Tc0w * priorTwc_) * Tcw_;
        cout << "predict pose diff with prior: " << diff << "\n";
    }
}

void KeyFrame::ReleaseMat() {
    grayImg_.release();
    debugGrayImg_.release();
}
