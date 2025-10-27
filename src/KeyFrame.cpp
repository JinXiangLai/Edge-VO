#include "KeyFrame.h"

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
KeyFrame::OpticalFlowStruct KeyFrame::optFlw;
std::mutex KeyFrame::mutexForSyncView3Dstatus;
std::unordered_set<KeyFrame*> KeyFrame::kfOn3Dshow;
std::shared_ptr<Camera> KeyFrame::cam_;
cv::Size KeyFrame::eachGridSize(0, 0);
cv::Ptr<cv::FastFeatureDetector> KeyFrame::detectorTh1, KeyFrame::detectorTh2;

class Landmark;

KeyFrame::KeyFrame(const cv::Mat& img, const Pose& Twc,
                   std::shared_ptr<Camera> cam, const int id, const int level)
    : id_(id),
      grayImg_(img),
      Twc_{Twc},
      Tcw_(Twc.Inverse()),
      priorTwc_(Twc),
      level_(level) {
    if (cam_ == nullptr) {
        cam_ = cam;
    }
    GenerateUndistordMap();
    cv::remap(grayImg_, grayImg_, map1, map2, cv::INTER_LINEAR);
    debugGrayImg_ = grayImg_.clone();
    CalculateEachGridForExtractFast();
    InitFastDetector();
}

KeyFrame::KeyFrame(const KeyFrame& f)
    : id_(f.id_),
      grayImg_(f.grayImg_),
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
    if (cam_ == nullptr) {
        cam_ = f.cam_;
    }

    debugGrayImg_ = f.debugGrayImg_;
    depthImage_ = f.depthImage_;
}

void KeyFrame::operator=(const KeyFrame& f) {
#if 1
    id_ = f.id_;
    grayImg_ = f.grayImg_;
    Twc_ = f.Twc_;
    Tcw_ = f.Tcw_;
    priorTwc_ = f.priorTwc_;
    level_ = f.level_;
    unKeypoints_ = f.unKeypoints_;
    convergeEdgeNum_ = f.convergeEdgeNum_;
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
    int deleteLKnum = 0;
    for (Landmark*& lk : landmark_) {
        if (lk != nullptr && lk->CanBeDelete() && lk->host_ == this) {
            delete lk;
            ++deleteLKnum;
            lk = nullptr;
        }
    }
    // ReleaseMat(); // 不需要手动释放
    if (invDepthUncertaintyFile_.is_open()) {
        invDepthUncertaintyFile_.close();
    }

    cout << fmt::format(
        "{} Release KF id: {}, landmark size: {}, delete size: {}, transform "
        "size: {}\n",
        reinterpret_cast<size_t>(this), id_, landmark_.size(), deleteLKnum,
        (landmark_.size() - deleteLKnum));
}

void KeyFrame::SetOpticalFlowStructCurFrame() {
    optFlw.prevImg_ = grayImg_;
    for (const auto& p : landmark_) {
        // 添加landmark对跟踪成功点的相互观测
        optFlw.prevPts_.emplace_back(cv::Point2f(p->uv_.x(), p->uv_.y()));
        optFlw.trackLandmark_.push_back(p);
    }
    cout << fmt::format("kf id: {}, SetOpticalFlowStructCurFrame num: {}\n",
                        id_, landmark_.size());
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

void KeyFrame::CalculateEachGridForExtractFast() {
    if (eachGridSize.height != 0) {
        return;
    }
    const float width = grayImg_.cols;
    const float height = grayImg_.rows;
    // 计算每个格子含有的像素个数
    const int gridPixelNum = static_cast<int>(
        width * height / config->extractFastNumEachFrame + 0.5);
    const float ratio = width / height;  // 宽高比
    const int gridHeight = static_cast<int>(sqrt(gridPixelNum / ratio) + 0.5);
    const int gridWidth = static_cast<int>(gridHeight * ratio + 0.5);
    cout << fmt::format(
        "gridHeight: {}, gridWidth: {}, need feature num: {}, recalculate "
        "feature num: {}\n",
        gridHeight, gridWidth, config->extractFastNumEachFrame,
        (width * height / (gridHeight * gridWidth)));
    constexpr int kMinGridHeight = 6;
    eachGridSize.width =
        max(static_cast<int>(kMinGridHeight * ratio + 0.5), gridWidth);
    eachGridSize.height = max(kMinGridHeight, gridHeight);
}

void KeyFrame::InitFastDetector() {
    if (detectorTh1 != nullptr) {
        return;
    }
    detectorTh1 = cv::FastFeatureDetector::create(
        config->fastTh1, true, cv::FastFeatureDetector::TYPE_9_16);

    detectorTh2 = cv::FastFeatureDetector::create(
        config->fastTh2, true, cv::FastFeatureDetector::TYPE_9_16);
}

int KeyFrame::RemoveNoInitializeLongFeature() {
    vector<Landmark*>::iterator it1 = optFlw.trackLandmark_.begin();
    vector<cv::Point2f>::iterator it2 = optFlw.prevPts_.begin();
    int removeFeatNum = 0;
    while (it1 != optFlw.trackLandmark_.begin() + optFlw.historyLandmarkNum_) {
        if ((*it1)->failInitializeNum_ > 1) {
            (*it1)->SetCanDelete();
            it1 = optFlw.trackLandmark_.erase(it1);
            it2 = optFlw.prevPts_.erase(it2);
            --optFlw.historyLandmarkNum_;
            ++removeFeatNum;
            continue;
        }
        ++it1;
        ++it2;
    }
    return removeFeatNum;
}

vector<cv::Point2f> KeyFrame::ExtractFastPointEachImage() {
    vector<cv::Point2f> res;
    res.reserve(config->extractFastNumEachFrame);
    for (int i = 0; i < grayImg_.rows; i += eachGridSize.height) {
        for (int j = 0; j < grayImg_.cols; j += eachGridSize.width) {
#if 0
            cv::Point2f fast(0, 0);
            if (ExtractFastPointEachGrid(i, j, config->fastTh1, fast)) {
                res.emplace_back(fast);
            } else if (ExtractFastPointEachGrid(i, j, config->fastTh2, fast)) {
                res.emplace_back(fast);
            }
#else
            const int w = min(eachGridSize.width, grayImg_.cols - j);
            const int h = min(eachGridSize.height, grayImg_.rows - i);
            const cv::Mat& gridImg = grayImg_(cv::Rect2i(j, i, w, h));
            vector<cv::KeyPoint> pts;
            detectorTh1->detect(gridImg, pts);
            if (pts.empty()) {
                detectorTh2->detect(gridImg, pts);
            }
            if (pts.empty()) {
                continue;
            }
            if (pts.size() > 1) {
                sort(pts.begin(), pts.end(),
                     [](const cv::KeyPoint& p1, const cv::KeyPoint& p2) {
                         return p1.response > p2.response;
                     });
            }
            // 只添加响应值最大的，但由于已经进行了极大值抑制，
            // res.emplace_back(j + pts[0].pt.x, i + pts[0].pt.y);
            // 故可以全部添加，影响不大
            for (const auto& p : pts) {
                res.emplace_back(j + p.pt.x, i + p.pt.y);
            }

#endif
        }
    }

    return res;
}

bool KeyFrame::ExtractFastPointEachGrid(const int diffRow, const int diffCol,
                                        const int fastTh1, cv::Point2f& fast) {
    int maxResponse = 0;
    for (int i = diffRow + 3; i < diffRow + eachGridSize.height; ++i) {
        for (int j = diffCol + 3; j < diffCol + eachGridSize.width; ++j) {
            int response = -1;
            IsFastPoint(grayImg_, fastTh1, {j, i}, response);
            if (response > maxResponse) {
                maxResponse = response;
                fast.x = static_cast<float>(j);
                fast.y = static_cast<float>(i);
            }
        }
    }

    return maxResponse > 0;
}

void KeyFrame::ExtractFastPoints(const OpticalFlowStruct& lastKFoptFlw) {
    cv::Mat search;
    if (!lastKFoptFlw.prevImg_.empty()) {
        search = cv::Mat::zeros(lastKFoptFlw.prevImg_.size(), CV_8UC1);
        // 当前关键帧追踪到当前帧的特征点，不要重复创建
        // 遍历当前帧被跟踪到的特征点

        auto SetNoGenerateKeypointArea = [&search](const cv::Point2f& p) {
            constexpr int windowLen = 10;
            constexpr int halfLen = windowLen / 2;
            constexpr int edgeLen = 2;
            const int tempTopY = static_cast<int>(p.y - halfLen);
            const int tempTopX = static_cast<int>(p.x - halfLen);
            const int topX = tempTopX < 0 ? edgeLen : tempTopX;
            const int topY = tempTopY < 0 ? edgeLen : tempTopY;
            const int downX = topX + windowLen > search.cols - 1
                                  ? search.cols - edgeLen
                                  : topX + windowLen;
            const int downY = topY + windowLen > search.rows - 1
                                  ? search.rows - edgeLen
                                  : topY + windowLen;
            if (downX <= topX || downY <= topY) {
                return;
            }
            search(cv::Rect(cv::Point(topX, topY), cv::Point(downX, downY)))
                .setTo(255);
        };
        for (const cv::Point2f& p : lastKFoptFlw.prevPts_) {
            SetNoGenerateKeypointArea(p);
        }
    }

    auto CanGenerateKeypoint = [&search](const cv::Point2f& p) -> bool {
        return search.empty() || search.ptr<uchar>(static_cast<int>(
                                     p.y))[static_cast<int>(p.x)] == 0;
    };

#if 0
    // 创建FAST检测器
    vector<cv::KeyPoint> pts;
    detectorTh1->detect(grayImg_, pts);
    pts.reserve(5000);
    unKeypoints_.reserve(pts.size());
    for (const cv::KeyPoint& p : pts) {
        // 注意：需要把上一KF的optFlw一直保留而不能重置
        if (search.empty() || CanGenerateKeypoint(p.pt)) {
            cv::circle(debugGrayImg_, p.pt, 2, kColor.at("white"));
            unKeypoints_.emplace_back(p.pt.x, p.pt.y);
        }
    }

#else
    vector<cv::Point2f> pts = ExtractFastPointEachImage();
    unKeypoints_.reserve(pts.size());
    for (const cv::Point2f& p : pts) {
        // 注意：需要把上一KF的optFlw一直保留而不能重置
        if (search.empty() || CanGenerateKeypoint(p)) {
            cv::circle(debugGrayImg_, p, 2, kColor.at("white"));
            unKeypoints_.emplace_back(p.x, p.y);
        }
    }
    cout << fmt::format(
        "self extract fast num: {}, current frame add kp num: {}, new create "
        "ratio: {:.2f}\n",
        pts.size(), unKeypoints_.size(),
        static_cast<double>(unKeypoints_.size()) / pts.size());
#endif
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

void KeyFrame::OpticalFlowTrackExecute(const cv::Mat& prevImg,
                                       const cv::Mat& curImg) {
    vector<cv::Point2f> nextPts;
    vector<uchar> status;
    vector<float> error;

    constexpr double kGoodMatchRatio = 0.99;
    auto GetGoodMatchMaxResidual = [&error]() -> float {
        vector<float> temp = error;
        sort(temp.begin(), temp.end());
        const int index = static_cast<int>(kGoodMatchRatio * temp.size());
        cout << fmt::format(
            "optflow track temp min error: {}, select error: {}, select +1 "
            "error: {},  max "
            "error: {}, temp size: "
            "{}\n",
            temp.front(), temp[index], temp[index + 1], temp.back(),
            temp.size());
        return temp[index];
    };

    const int winLen = config->optflowWinSize;
    cv::calcOpticalFlowPyrLK(prevImg, curImg, optFlw.prevPts_, nextPts, status,
                             error, cv::Size(winLen, winLen),
                             config->optflowLayer);
    // 使用min会导致有效跟踪逐渐减少，导致关键帧频繁更新，最终影响系统的精度，甚至失败
    const float maxError = max(static_cast<float>(config->maxFlowTrackError),
                               GetGoodMatchMaxResidual());
    cout << fmt::format(
        "adaptive optflow track maxError: {}, onfig->maxFlowTrackError: {}\n",
        maxError, config->maxFlowTrackError);

#if defined(WRITE_MATCH_PAIR_IMAGE)
    const auto debugPts1 = optFlw.prevPts_;
#endif

    optFlw.prevPts_.clear();
    vector<Landmark*> trackLandmark;
    size_t historyLandmarkTrackSuccessNum = 0;
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i] == 1 && error[i] < maxError) {
            // 重新赋值landmark在当前帧上的观测
            optFlw.prevPts_.emplace_back(nextPts[i]);
            trackLandmark.emplace_back(optFlw.trackLandmark_[i]);
            if (i < optFlw.historyLandmarkNum_) {
                ++historyLandmarkTrackSuccessNum;
            }

#if defined(WRITE_MATCH_PAIR_IMAGE)
            Eigen::Vector2i p1(int(debugPts1[i].x), int(debugPts1[i].y));
            Eigen::Vector2i p2(int(nextPts[i].x), int(nextPts[i].y));
            DrawBestMatchEachFrame(p1, p2, curImg, true);
#endif
        }
    }

    optFlw.historyLandmarkNum_ = historyLandmarkTrackSuccessNum;
    optFlw.trackLandmark_ = std::move(trackLandmark);

    // 跟踪成功后，重新赋值
    cout << fmt::format(
        "optical flow tracked info: track last KF landmark num: {}, "
        "track history landmark num: {}\n",
        optFlw.trackLandmark_.size() - historyLandmarkTrackSuccessNum,
        historyLandmarkTrackSuccessNum);
}

void KeyFrame::OpticalFlowTrackLandmark(const KeyFrame& f2) {

    if (optFlw.prevPts_.empty()) {
        cout << "here optFlw_.prevPts_ should not be empty!!!\n";
        exit(-1);
    }

    // 上一关键帧对当前帧的跟踪结果
    OpticalFlowTrackExecute(optFlw.prevImg_, f2.grayImg_);

#if defined(WRITE_MATCH_PAIR_IMAGE)
    WriteDebugImage2VideoEachFrame(f2.id_, "lastKF_track_result.avi");
#endif

    optFlw.prevImg_ = f2.grayImg_;
}

double KeyFrame::TrackWithOpticalFlow(const KeyFrame& kf2, int& findMatchNum) {
    OpticalFlowTrackLandmark(kf2);
    findMatchNum = optFlw.GetTrackFeatureNum();
    return optFlw.GetTrackFeatureRatio();
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
            cvtColor(optFlw.prevImg_, im1, cv::COLOR_GRAY2BGR);
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
    const OpticalFlowStruct lastKFoptFlw =
        lastKf == nullptr ? OpticalFlowStruct() : optFlw;
    if (landmark_.empty()) {
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

    //把上一关键帧中保留的光流及历史关键帧的光流跟踪结果合并到当前关键帧
    if (lastKf != nullptr) {
        optFlw.historyLandmarkNum_ = optFlw.prevPts_.size();
    }

    // 初始化当前新建关键帧进行光流跟踪所需的结构，仅针对当前KF
    SetOpticalFlowStructCurFrame();

    optFlw.SetTotalFeatureCreated();

    // 历史关键帧和当前关键帧在当前灰度图上提取到的关键点数量
    return optFlw.GetTrackFeatureNum();
}

void KeyFrame::Update(const Eigen::Vector3d& delta_q,
                      const Eigen::Vector3d& delta_t) {
    Twc_.Update(delta_q, delta_t);
    Tcw_ = Twc_.Inverse();
}

void KeyFrame::SetTwc(const Pose& Twc, const bool printDiff) {
    if (printDiff && Tc0w.t_wb_.isApproxToConstant(0)) {
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
