#include "KeyFrame.h"

#include <cstddef>
#include <cstdint>
#include <flann/flann.hpp>
#include <opencv2/highgui.hpp>
#include <random>
#include <string>
#include <thread>
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
std::mutex KeyFrame::mutexForSyncView3Dstatus;
std::unordered_set<KeyFrame*> KeyFrame::kfOn3Dshow;
std::shared_ptr<Camera> KeyFrame::cam_;
cv::Size KeyFrame::eachGridSize(0, 0);
cv::Ptr<cv::FastFeatureDetector> KeyFrame::detectorTh1, KeyFrame::detectorTh2;
ofstream KeyFrame::poseFile;
ofstream KeyFrame::kfPoseFile;
string KeyFrame::poseFilePath, KeyFrame::kfPoseFilePath;
vector<pair<double, string>> KeyFrame::vecTime2Pose;
cv::Mat KeyFrame::superpointLocation;

class Landmark;

KeyFrame::KeyFrame(const cv::Mat& img, const Pose& Twc, shared_ptr<Camera> cam,
                   const int id, const double timestamp, const int level)
    : id_(id),
      grayImg_(img),
      Twc_{Twc},
      Tcw_(Twc.Inverse()),
      priorTwc_(Twc),
      level_(level),
      timestamp_(timestamp) {
    if (cam_ == nullptr) {
        cam_ = cam;
    }
    GenerateUndistordMap();
    cv::remap(grayImg_, grayImg_, map1, map2, cv::INTER_LINEAR);
    debugGrayImg_ = grayImg_.clone();
    CalculateEachGridForExtractFast();

    InitSuperpointAndLightglueEngine();

    InitFastDetector();

    SetBackupPose();
}

KeyFrame::KeyFrame(const KeyFrame& f)
    : id_(f.id_),
      grayImg_(f.grayImg_),
      Twc_(f.Twc_),
      Tcw_(f.Tcw_),
      priorTwc_(f.priorTwc_),
      level_(f.level_),
      outOfRange_(f.outOfRange_),
      convergeEdgeNum_(f.convergeEdgeNum_),
      kpts_(f.kpts_),
      desc_(f.desc_),
      timestamp_(f.timestamp_) {
    // vector内的堆内存需要先释放
    // 不能这样子，这是构造函数，默认的内存应该是干净的，
    // 否则你应该调用赋值构造
    if (cam_ == nullptr) {
        cam_ = f.cam_;
    }

    debugGrayImg_ = f.debugGrayImg_;
    depthImage_ = f.depthImage_;
    SetBackupPose();
}

void KeyFrame::operator=(const KeyFrame& f) {
#if 1
    id_ = f.id_;
    grayImg_ = f.grayImg_;
    Twc_ = f.Twc_;
    Tcw_ = f.Tcw_;
    priorTwc_ = f.priorTwc_;
    level_ = f.level_;
    convergeEdgeNum_ = f.convergeEdgeNum_;
    timestamp_ = f.timestamp_;
    kpts_ = f.kpts_;
    desc_ = f.desc_;
#else
    ReleaseMat();
    // 这样会导致cv::Mat等堆内存无法释放
    new (this) KeyFrame(f);
#endif
    depthImage_ = f.depthImage_;
    debugGrayImg_ = f.debugGrayImg_;
    SetBackupPose();
}

KeyFrame::~KeyFrame() {
    // 由于Landmar与KeyFrame相互引用，所以之前将析构函数放在头文件导致landmark_内存无法释放？？
    // int deleteLKnum = 0;
    // for (Landmark*& lk : landmark_) {
    //     if (lk != nullptr && lk->CanBeDelete() && lk->host_ == this) {
    //         // 同步将其余观测的指针置空，避免悬空，
    //         // 因为滑窗内其余观测到该Landmark*的Keyframe*中的landmar_数组，也会存储该Landmark*
    //         for (auto& p : lk->target_) {
    //             p.first->landmark_[p.second] = nullptr;
    //         }
    //         delete lk;
    //         ++deleteLKnum;
    //         lk = nullptr;
    //     }
    // }
    // ReleaseMat(); // 不需要手动释放
    if (invDepthUncertaintyFile_.is_open()) {
        invDepthUncertaintyFile_.close();
    }

    // cout << fmt::format(
    //     "{} Release KF id: {}, landmark size: {}, delete size: {}, transform "
    //     "size: {}\n",
    //     reinterpret_cast<size_t>(this), id_, landmark_.size(), deleteLKnum,
    //     (landmark_.size() - deleteLKnum));
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
#if USE_SUPERPOINT_AND_LIGHTGLUE
    const int featureNum = SuperPointConfig::kMaxKeypoints;
#else
    const int featureNum = config->extractFastNumEachFrame;
#endif
    // 计算每个格子含有的像素个数
    const int gridPixelNum =
        static_cast<int>(width * height / featureNum + 0.5);
    const float ratio = width / height;  // 宽高比
    const int gridHeight = static_cast<int>(sqrt(gridPixelNum / ratio) + 0.5);
    const int gridWidth = static_cast<int>(gridHeight * ratio + 0.5);
    cout << fmt::format(
        "gridHeight: {}, gridWidth: {}, need feature num: {}, recalculate "
        "feature num: {}\n",
        gridHeight, gridWidth, featureNum,
        (width * height / (gridHeight * gridWidth)));
    constexpr int kMinGridHeight = 6;
    eachGridSize.width =
        max(static_cast<int>(kMinGridHeight * ratio + 0.5), gridWidth);
    eachGridSize.height = max(kMinGridHeight, gridHeight);
}

void KeyFrame::InitSuperpointAndLightglueEngine() {
    if (superpointPtr != nullptr && lightgluePtr != nullptr) {
        return;
    }

    superpointPtr = make_shared<SuperPoint>(config->superpointOnnxFilePath,
                                            config->superpointEngineFilePath);
    superpointPtr->SetEachGridSize2ExtractOnePoint(eachGridSize);
    if (!superpointPtr->Build()) {
        cerr << "Error in SuperPoint building engine. Please check your "
                "onnx model path."
             << endl;
    }

    lightgluePtr = make_shared<LightGlue>(config->lightglueOnnxFilePath,
                                          config->lightglueEngineFilePath);
    if (!lightgluePtr->Build()) {
        cerr << "Error in lightglue building engine. Please check your "
                "onnx model path."
             << endl;
    }
    lightgluePtr->SetThreshold(LightGlueConfig::kMatchThreshold);
    cout << "SuperPoint and lightglue inference engine build success." << endl;
    lightgluePtr->ValidateFP16();

    superpointPtr->WarmUp();
    lightgluePtr->WarmUp();
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
    vector<shared_ptr<Landmark>>::iterator it1 =
        globalOptFlw.trackLandmark_.begin();
    vector<cv::Point2f>::iterator it2 = globalOptFlw.prevPts_.begin();
    int removeFeatNum = 0;
    constexpr int kMaxNotInitSuccessNum = 10;
    while (it1 != globalOptFlw.trackLandmark_.begin() +
                      globalOptFlw.historyLandmarkNum_) {
        if ((*it1)->failInitializeNum_ >= kMaxNotInitSuccessNum) {
            (*it1)->SetCanDelete();
            it1 = globalOptFlw.trackLandmark_.erase(it1);
            it2 = globalOptFlw.prevPts_.erase(it2);
            --globalOptFlw.historyLandmarkNum_;
            ++removeFeatNum;
            continue;
        }
        ++it1;
        ++it2;
    }
    return removeFeatNum;
}

void KeyFrame::ExtractSuperpoint() {
    chrono::steady_clock::time_point t0 = chrono::steady_clock::now();
    if (!superpointPtr->Infer(grayImg_, kpts_, desc_)) {
        cerr << "Failed when extracting features from first image." << endl;
    } else {
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

        cout << fmt::format(
            "Superpoint extract {} points, total spend: {:.1f}ms\n",
            kpts_.rows(), ChronoMillisecTimeDuration(t0, t1));
        for (int i = 0; i < kpts_.rows(); ++i) {
            const cv::Point2f p(kpts_(i, 0), kpts_(i, 1));
            cv::circle(debugGrayImg_, p, 2, kColor.at("orange"));
        }
    }
}

void KeyFrame::ExtractFastPoints() {
    // 提取当前KF的关键点，上一光流跟踪结果在当前KF必须是关键点，否则设为delete
    vector<cv::Point2f> pts = ExtractFastPointEachGridImage();
    cv::Mat keypointInCurImg;
    // 锁住全局光流变量，只访问不修改
    lock_guard<mutex> lock(globalOptFlwMutex);
    // 既然光流跟踪成功，这里就不应该再限制删除跟踪成功的点
    if (0 && !globalOptFlw.prevPts_.empty()) {
        keypointInCurImg = cv::Mat::zeros(grayImg_.size(), CV_8UC1);

        auto SetCurImgKeypointArea = [&keypointInCurImg](const cv::Point2f& p) {
            constexpr int windowLen = 6;
            constexpr int halfLen = windowLen / 2;
            constexpr int edgeLen = 2;  // 无效边缘点
            const int tempTopY = static_cast<int>(p.y - halfLen);
            const int tempTopX = static_cast<int>(p.x - halfLen);
            const int topX = tempTopX < 0 ? edgeLen : tempTopX;
            const int topY = tempTopY < 0 ? edgeLen : tempTopY;
            const int downX = topX + windowLen > keypointInCurImg.cols - 1
                                  ? keypointInCurImg.cols - edgeLen
                                  : topX + windowLen;
            const int downY = topY + windowLen > keypointInCurImg.rows - 1
                                  ? keypointInCurImg.rows - edgeLen
                                  : topY + windowLen;
            if (downX <= topX || downY <= topY) {
                return;
            }
            keypointInCurImg(
                cv::Rect(cv::Point(topX, topY), cv::Point(downX, downY)))
                .setTo(255);
        };

        for (const cv::Point2f& p : pts) {
            SetCurImgKeypointArea(p);
        }

        for (size_t i = 0; i < globalOptFlw.trackLandmark_.size(); ++i) {
            const cv::Point2f& p = globalOptFlw.prevPts_[i];
            if (keypointInCurImg.ptr<uchar>(
                    static_cast<int>(p.y))[static_cast<int>(p.x)] == 0) {
                // 在当前帧跟踪错误，不是角点了，因此设置为删除
                globalOptFlw.trackLandmark_[i]->SetCanDelete();
            }
        }
    }

    // 移除掉无效的landmark*，为后续创建关键点提供便利
    globalOptFlw.RemoveUselessLandmark();

    cv::Mat search;
    if (!globalOptFlw.prevImg_.empty()) {
        search = cv::Mat::zeros(globalOptFlw.prevImg_.size(), CV_8UC1);
        // 当前关键帧追踪到当前帧的特征点，不要重复创建
        // 遍历当前帧被跟踪到的特征点
        auto SetNoGenerateKeypointArea = [&search](const cv::Point2f& p) {
            // 既然是当前帧提取的大响应值点，那么就应该尽力让它被选择，但会额外引入更多landmark点
            constexpr int windowLen = 16;
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
        for (size_t i = 0; i < globalOptFlw.prevPts_.size(); ++i) {
            SetNoGenerateKeypointArea(globalOptFlw.prevPts_[i]);
        }
    }

    auto CanGenerateKeypoint = [&search](const cv::Point2f& p) -> bool {
        return search.empty() || search.ptr<uchar>(static_cast<int>(
                                     p.y))[static_cast<int>(p.x)] == 0;
    };

    vector<size_t> newPtsIdx;
    newPtsIdx.reserve(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) {
        const cv::Point2f& p = pts[i];
        // 注意：需要把上一KF的optical flow一直保留而不能重置
        if (search.empty() || CanGenerateKeypoint(p)) {
            cv::circle(debugGrayImg_, p, 2, kColor.at("white"));
            newPtsIdx.emplace_back(i);
        }
    }

    // 需要把之前帧在当前帧的匹配特征点加上，注意，此时已经移除optflw中所有无效Landmark*
    const int fastPointNum = newPtsIdx.size() + globalOptFlw.prevPts_.size();
    kpts_.conservativeResize(desc_.rows() + fastPointNum, 2);
    //kpts_.resize(newPtsIdx.size() + globalOptFlw.prevPts_.size(), 2);
    int startRow = desc_.rows();
    for (size_t i = 0; i < globalOptFlw.prevPts_.size(); ++i) {
        const cv::Point2f& p = globalOptFlw.prevPts_[i];
        kpts_.row(startRow + i) << p.x, p.y;
    }

    startRow += int(globalOptFlw.prevPts_.size());
    for (size_t i = 0; i < newPtsIdx.size(); ++i) {
        const cv::Point2f& p = pts[newPtsIdx[i]];
        kpts_.row(startRow + i) << p.x, p.y;
    }

    cout << fmt::format(
        "cur kf extract fast num: {}, current frame add kp num: {}, new create "
        "ratio: {:.2f}\n",
        pts.size(), newPtsIdx.size(),
        static_cast<double>(newPtsIdx.size()) / kpts_.rows());
}

vector<cv::Point2f> KeyFrame::ExtractFastPointEachGridImage() {
    if (superpointLocation.empty()) {
        superpointLocation = cv::Mat::zeros(grayImg_.size(), 0);
    } else {
        superpointLocation.setTo(0);
    }
    for (int i = 0; i < kpts_.rows(); ++i) {
        const auto& p = kpts_.row(i);
        superpointLocation.ptr<uchar>(int(p[1]))[int(p[0])] = 1;
    }
    vector<cv::Point2f> res;
    res.reserve(config->extractFastNumEachFrame);
    constexpr int kNoSuperpointRangeLen = 7;  // 须是奇数
    constexpr int kDiff = kNoSuperpointRangeLen / 2;
    for (int i = 0; i < grayImg_.rows; i += eachGridSize.height) {
        for (int j = 0; j < grayImg_.cols; j += eachGridSize.width) {

            const int w = min(eachGridSize.width, grayImg_.cols - j);
            const int h = min(eachGridSize.height, grayImg_.rows - i);
            if (cv::countNonZero(superpointLocation(cv::Rect2i(j, i, w, h)))) {
                continue;
            }
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
            if (pts.size() > 1) {
                res.emplace_back(j + pts[0].pt.x, i + pts[0].pt.y);
                res.emplace_back(j + pts[1].pt.x, i + pts[1].pt.y);
            } else {
                res.emplace_back(j + pts[0].pt.x, i + pts[0].pt.y);
            }
            // 故可以全部添加，影响不大
            // for (const auto& p : pts) {
            //     if (!cv::countNonZero(superpointLocation(cv::Rect2i(
            //             p.pt.x - kDiff, p.pt.y - kDiff, kNoSuperpointRangeLen,
            //             kNoSuperpointRangeLen)))) {
            //         res.emplace_back(j + p.pt.x, i + p.pt.y);
            //     }
            // }
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

void KeyFrame::ExtractFeaturetPoints() {

#ifdef USE_SUPERPOINT_AND_LIGHTGLUE
    ExtractSuperpoint();
#else
    ExtractSuperpoint();
    ExtractFastPoints();
#endif

    // 为每个提取到的角点生成一个Landmark对象，但不在这里申请内存，避免后续无法释放跟踪成功landmark的内存
    landmark_.resize(kpts_.rows(), nullptr);
    // for (int i = 0; i < kpts_.rows(); ++i) {
    //     // shared_ptr<KeyFrame>(this)会导致多源智能指针，它会释放多次KeyFrame导致报错
    //     // 若需要使用智能指针，必须保证this在此前已经由一个智能指针管理，然后使用shared_from_this()来获取，否则只能使用原始指针
    //     // landmark_.push_back(make_shared<Landmark>(upx, make_shared<KeyFrame>(this), cam_, 1.0) ); [ERROR double free]

    //     landmark_[i] = new Landmark(i, this, cam_, kInitInvDepth);
    // }
}

int KeyFrame::LightglueMatchAndRefineTrackResult(KeyFrame* lastKf) {
    if (lastKf == nullptr) {
        lock_guard<mutex> lock(globalOptFlwMutex);
        // 初始化世界帧
        for (int i = 0; i < kpts_.rows(); ++i) {
            landmark_[i] = make_shared<Landmark>(i, this, cam_, kInitInvDepth);
            if (config->useDepthImage) {
                int x = static_cast<int>(kpts_.row(i)[0]);
                int y = static_cast<int>(kpts_.row(i)[1]);
                double d = static_cast<double>(depthImage_.ptr<ushort>(y)[x]);
                landmark_[i]->trueDepth_ = d > 0 ? d /= config->depthFactor : 0;
            }
            globalOptFlw.trackLandmark_.emplace_back(landmark_[i]);
            globalOptFlw.prevPts_.emplace_back(kpts_(i, 0), kpts_(i, 1));
        }
        globalOptFlw.totalFeatureCreated_ = globalOptFlw.trackLandmark_.size();
        globalOptFlw.prevImg_ = grayImg_;
        return globalOptFlw.prevPts_.size();
    }

#ifdef USE_SUPERPOINT_AND_LIGHTGLUE
    // 当前帧反追踪上一帧
    Eigen::VectorXf mscores;
    vector<cv::DMatch> lightglueMatches;
    int matchPairNum = 0;
    chrono::steady_clock::time_point t0 = chrono::steady_clock::now();
    if (true || globalOptFlw.historyLandmarkNum_ == 0) {
        matchPairNum = lightgluePtr->MatchKeypoints(kpts_, lastKf->kpts_, desc_,
                                                    lastKf->desc_, mscores,
                                                    lightglueMatches);
    } else {
        SupperPointMatch(lastKf, lightglueMatches);
        vector<cv::DMatch> lightglueMatches2;
        lastKf->SupperPointMatch(this, lightglueMatches2);
        for (int i = 0; i < lightglueMatches.size(); ++i) {
            if (lightglueMatches[i].trainIdx < 0) {
                continue;
            }
            if (lightglueMatches2[lightglueMatches[i].trainIdx].trainIdx == i) {
                lightglueMatches[matchPairNum++] = lightglueMatches[i];
            }
        }
        lightglueMatches.resize(matchPairNum);
    }
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

    cout << fmt::format(
        "kf id: {}, last_kf id: {}, matchPairNum: {}, match ratio: {:.1f}, "
        "spend: {:.1f}ms, lightglue used: {}\n",
        id_, lastKf->id_, matchPairNum, double(matchPairNum) / kpts_.rows(),
        ChronoMillisecTimeDuration(t0, t1),
        globalOptFlw.historyLandmarkNum_ == 0);

    // 可视化匹配结果
    if (0) {
        const cv::Mat& image0 = grayImg_;
        const cv::Mat& image1 = lastKf->grayImg_;
        const auto &kpts0 = kpts_, &kpts1 = lastKf->kpts_;
        cv::Mat matchImage = cv::Mat(image0.rows, image0.cols * 2, CV_8UC1);
        cv::Mat matchImgColor = cv::Mat(image0.rows, image0.cols * 2, CV_8UC3);
        auto GetRandColor = []() -> cv::Scalar_<int> {
            return {abs(rand()) % 256, abs(rand()) % 256, abs(rand()) % 256};
        };

        image0.copyTo(matchImage(cv::Rect(0, 0, image0.cols, image0.rows)));
        image1.copyTo(
            matchImage(cv::Rect(image0.cols, 0, image0.cols, image0.rows)));
        cv::cvtColor(matchImage, matchImgColor, cv::COLOR_GRAY2BGR);
        // lightglueMatches指示了哪些点应该有连线
        for (const cv::DMatch m : lightglueMatches) {
            const cv::Scalar bgr = GetRandColor();
            const int i = m.queryIdx;
            const int j = m.trainIdx;
            const cv::Point2f p1(kpts0(i, 0), kpts0(i, 1));
            const cv::Point2f p2(kpts1(j, 0) + image0.cols, kpts1(j, 1));
            cv::circle(matchImgColor, p1, 2, bgr);
            cv::circle(matchImgColor, p2, 2, bgr);
            cv::line(matchImgColor, p1, p2, bgr);
        }

        cv::putText(matchImgColor, fmt::format("match num: {}", matchPairNum),
                    cv::Point(10, 30), cv::FONT_ITALIC, 0.80, {0, 0, 255}, 2);

        // cv::imwrite("matchImage.png", matchImage);
        //  visualize
        cv::imshow("matchImgColor", matchImgColor);
        cv::waitKey(0);
    }

    // 清除上一KF的光流跟踪结果，使用lightglue匹配的结果替换，未匹配上的直接丢弃
    // Landmark*将会保留历史跟踪结果target，这里直接对optflw进行重新赋值，然后用于三角化即可
    OpticalFlowStruct glueMatch;
    vector<bool> trackedKpId(kpts_.rows(), false);
    for (const cv::DMatch m : lightglueMatches) {
        const int curId = m.queryIdx;
        const int lastId = m.trainIdx;
        if (lastKf->landmark_[lastId] == nullptr ||
            lastKf->landmark_[lastId]->CanBeDelete()) {
            continue;
        }

        // 与上一帧匹配的，直接使用上一帧有效的Landmark*修改当前帧的landmark*，
        landmark_[curId] = lastKf->landmark_[lastId];
        // 添加新的相互观测
        landmark_[curId]->AddNewKFobservation(this, curId);
        trackedKpId[curId] = true;

        // 光流跟踪使用，需要记录该landmark*对应当前帧的关键点位置
        glueMatch.trackLandmark_.emplace_back(landmark_[curId]);
        glueMatch.prevPts_.emplace_back(kpts_(curId, 0), kpts_(curId, 1));
    }
    glueMatch.historyLandmarkNum_ = glueMatch.trackLandmark_.size();
    cout << fmt::format("kf id: {}, track history landmark num: {}\n", id_,
                        glueMatch.historyLandmarkNum_);

    // 当前关键帧新增关键点也加入光流跟踪，虽不能三角化，但可用于判断视角变化
    for (size_t i = 0; i < trackedKpId.size(); ++i) {
        if (trackedKpId[i]) {
            continue;
        }
        landmark_[i] = make_shared<Landmark>(i, this, cam_, kInitInvDepth);
        if (config->useDepthImage) {
            int x = static_cast<int>(kpts_.row(i)[0]);
            int y = static_cast<int>(kpts_.row(i)[1]);
            double d = static_cast<double>(depthImage_.ptr<ushort>(y)[x]);
            landmark_[i]->trueDepth_ = d > 0 ? d /= config->depthFactor : 0;
        }
        glueMatch.trackLandmark_.emplace_back(landmark_[i]);
        glueMatch.prevPts_.emplace_back(kpts_(i, 0), kpts_(i, 1));
    }
    glueMatch.totalFeatureCreated_ = glueMatch.trackLandmark_.size();
    const int newAddLandmarkNum =
        glueMatch.totalFeatureCreated_ - glueMatch.historyLandmarkNum_;
    cout << fmt::format(
        "cur kf id: {}, track history landmark num: {}, new add landmark num: "
        "{}, add landmark ratio: {:.1f}\n",
        id_, glueMatch.historyLandmarkNum_, newAddLandmarkNum,
        double(newAddLandmarkNum) / glueMatch.totalFeatureCreated_);

    {
        lock_guard<mutex> lock(globalOptFlwMutex);
        globalOptFlw = std::move(glueMatch);
        globalOptFlw.prevImg_ = grayImg_;
    }

    return matchPairNum;

#else
    // 不使用lightglue进行匹配，以比较效果
    //把上一关键帧中保留的光流及历史关键帧的光流跟踪结果合并到当前关键帧
    // 移除掉所有无效Landmark*，这里应该在提取当前帧的关键点时就要调用
    {
        lock_guard<mutex> lock(globalOptFlwMutex);
        globalOptFlw.historyLandmarkNum_ = globalOptFlw.prevPts_.size();
        // 历史关键点在当前帧的跟踪结果需要进行相互观测赋值
        int startRow = desc_.rows();
        for (size_t i = 0; i < globalOptFlw.prevPts_.size(); ++i) {
            const int row = startRow + i;
            landmark_[row] = globalOptFlw.trackLandmark_[i];
            landmark_[row]->AddNewKFobservation(this, row);
        }

        // 初始化当前新建关键帧进行光流跟踪所需的结构，仅针对当前KF
        startRow += int(globalOptFlw.prevPts_.size());
        for (int i = startRow; i < kpts_.rows(); ++i) {
            // 当前帧新提取的关键帧加入结果
            landmark_[i] = make_shared<Landmark>(i, this, cam_, kInitInvDepth);
            if (config->useDepthImage) {
                int x = static_cast<int>(kpts_.row(i)[0]);
                int y = static_cast<int>(kpts_.row(i)[1]);
                double d = static_cast<double>(depthImage_.ptr<ushort>(y)[x]);
                landmark_[i]->trueDepth_ = d > 0 ? d /= config->depthFactor : 0;
            }
            globalOptFlw.trackLandmark_.emplace_back(landmark_[i]);
            globalOptFlw.prevPts_.emplace_back(kpts_(i, 0), kpts_(i, 1));
        }

        for (int i = 0; i < desc_.rows(); ++i) {
            landmark_[i] = make_shared<Landmark>(i, this, cam_, kInitInvDepth);
            if (config->useDepthImage) {
                int x = static_cast<int>(kpts_.row(i)[0]);
                int y = static_cast<int>(kpts_.row(i)[1]);
                double d = static_cast<double>(depthImage_.ptr<ushort>(y)[x]);
                landmark_[i]->trueDepth_ = d > 0 ? d /= config->depthFactor : 0;
            }
            globalOptFlw.trackLandmark_.emplace_back(landmark_[i]);
            globalOptFlw.prevPts_.emplace_back(kpts_(i, 0), kpts_(i, 1));
        }

        globalOptFlw.SetTotalFeatureCreated();
    }

    // 历史关键帧和当前关键帧在当前灰度图上提取到的关键点数量
    return globalOptFlw.GetTrackFeatureNum();
#endif
}

void KeyFrame::AddReportElement(const string& key) {
    if (matchResultStatiscs_.count(key)) {
        matchResultStatiscs_[key]++;
    } else {
        matchResultStatiscs_[key] = 1;
    }
}

void KeyFrame::ResetDebugMessage() {
    matchResultStatiscs_.clear();
}

string KeyFrame::OutputPoseMessage() const {
    const Eigen::Vector3d& p = Twc_.t_wb_;
    const Eigen::Quaterniond& q = Twc_.q_wb_;
    return fmt::format("{:.6f} {} {} {} {} {} {} {}", timestamp_, p.x(), p.y(),
                       p.z(), q.x(), q.y(), q.z(), q.w());
}

void KeyFrame::OpticalFlowTrackExecute(const cv::Mat& prevImg,
                                       const cv::Mat& curImg) {
    vector<cv::Point2f> nextPts;
    vector<uchar> status;
    vector<float> error;

    constexpr double kGoodMatchRatio = 0.75;
    auto GetGoodMatchMaxResidual = [&error, &status]() -> float {
        vector<float> temp;
        temp.reserve(error.size());
        for (size_t i = 0; i < error.size(); ++i) {
            if (status[i] == 1) {
                temp.emplace_back(error[i]);
            }
        }
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
    cout << fmt::format(
                "debug prevImg size: [{}x{}], curImg size: [{}x{}], "
                "globalOptFlw.prevPts_.size: {}, nextPts.size: {}",
                prevImg.rows, prevImg.cols, curImg.rows, curImg.cols,
                globalOptFlw.prevPts_.size(), nextPts.size())
         << endl;
    cv::calcOpticalFlowPyrLK(prevImg, curImg, globalOptFlw.prevPts_, nextPts,
                             status, error, cv::Size(winLen, winLen),
                             config->optflowLayer);
    // 使用min会导致有效跟踪逐渐减少，导致关键帧频繁更新，最终影响系统的精度，甚至失败
    const float maxError = max(static_cast<float>(config->maxFlowTrackError),
                               GetGoodMatchMaxResidual());
    cout << fmt::format(
        "adaptive optflow track maxError: {}, onfig->maxFlowTrackError: {}\n",
        maxError, config->maxFlowTrackError);

#if defined(WRITE_MATCH_PAIR_IMAGE)
    const auto debugPts1 = globalOptFlw.prevPts_;
#endif

    vector<shared_ptr<Landmark>> trackLandmark;
    size_t historyLandmarkTrackSuccessNum = 0;
    vector<double> parallaxVec;
    parallaxVec.reserve(500);
    const Eigen::Vector2d mainPoint(cam_->cx_, cam_->cy_);

    {
        //lock_guard<mutex> lock(globalOptFlwMutex); // 调用处加锁
        globalOptFlw.prevPts_.clear();
        for (size_t i = 0; i < status.size(); ++i) {
            if (status[i] == 1 && error[i] < maxError) {
                // 重新赋值landmark在当前帧上的观测
                globalOptFlw.prevPts_.emplace_back(nextPts[i]);
                trackLandmark.emplace_back(globalOptFlw.trackLandmark_[i]);
                if (i < globalOptFlw.historyLandmarkNum_) {
                    ++historyLandmarkTrackSuccessNum;
                } else {
                    const double parallax = CalculateParallax(
                        globalOptFlw.trackLandmark_[i]->GetHostFrameObv(),
                        {nextPts[i].x, nextPts[i].y}, mainPoint);
                    if (parallax > 0.) {
                        parallaxVec.emplace_back(parallax);
                    }
                }

#if defined(WRITE_MATCH_PAIR_IMAGE)
                Eigen::Vector2i p1(int(debugPts1[i].x), int(debugPts1[i].y));
                Eigen::Vector2i p2(int(nextPts[i].x), int(nextPts[i].y));
                DrawBestMatchEachFrame(p1, p2, curImg, true);
#endif
            }
        }

        globalOptFlw.historyLandmarkNum_ = historyLandmarkTrackSuccessNum;
        globalOptFlw.trackLandmark_ = std::move(trackLandmark);

        sort(parallaxVec.begin(), parallaxVec.end());
        globalOptFlw.meanParallax_ =
            parallaxVec[static_cast<int>(parallaxVec.size() * 0.1)];

        globalOptFlw.prevImg_ = curImg;
        // 跟踪成功后，重新赋值
        cout << fmt::format(
            "optical flow tracked info: track last KF landmark num: {}, "
            "track history landmark num: {}, parallax since last KF: {:.1f}, "
            "usefulParallaxNum: {}\n",
            globalOptFlw.trackLandmark_.size() - historyLandmarkTrackSuccessNum,
            historyLandmarkTrackSuccessNum, globalOptFlw.meanParallax_,
            parallaxVec.size());
    }
}

void KeyFrame::OpticalFlowTrackLandmark(const KeyFrame& f2) {

    {
        //lock_guard<mutex> lock(globalOptFlwMutex); // 调用处加锁

        globalOptFlw.RemoveUselessLandmark();

        if (globalOptFlw.prevPts_.empty()) {
            cout << "here globalOptFlw.prevPts_ should not be empty!!!\n";
            exit(-1);
        }
    }

    // 上一关键帧对当前帧的跟踪结果
    OpticalFlowTrackExecute(globalOptFlw.prevImg_, f2.grayImg_);

#if defined(WRITE_MATCH_PAIR_IMAGE)
    WriteDebugImage2VideoEachFrame(f2.id_, "lastKF_track_result.avi");
#endif
}

double KeyFrame::TrackWithOpticalFlow(const KeyFrame& kf2, int& findMatchNum) {
    lock_guard<mutex> lock(globalOptFlwMutex);
    OpticalFlowTrackLandmark(kf2);
    findMatchNum = globalOptFlw.GetTrackFeatureNum();
    return globalOptFlw.GetTrackFeatureRatio();
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

void KeyFrame::InitPoseFileMessage() {
    if (poseFile.is_open()) {
        poseFile.close();
    }
    if (kfPoseFile.is_open()) {
        kfPoseFile.close();
    }
    poseFilePath = fmt::format(
        "{}/{}.txt", config->debugMessageSaveFolder,
        config->dataDir.substr(config->dataDir.find_last_of('/') + 1));
    kfPoseFilePath = fmt::format(
        "{}/{}_kf.txt", config->debugMessageSaveFolder,
        config->dataDir.substr(config->dataDir.find_last_of('/') + 1));
    poseFile.open(poseFilePath.c_str(), ios::out);
    kfPoseFile.open(kfPoseFilePath.c_str(), ios::out);
    if (!poseFile.is_open()) {
        cerr << fmt::format("Open {} file failed!\n", poseFilePath);
        exit(-1);
    }
    if (!kfPoseFile.is_open()) {
        cerr << fmt::format("Open {} file failed!\n", kfPoseFilePath);
        exit(-1);
    }
    vecTime2Pose.clear();
    vecTime2Pose.reserve(500);
}

void KeyFrame::WritePoseMessage2File(const KeyFrame& f) {
    if (!f.landmark_.empty()) {
        vecTime2Pose.emplace_back(f.timestamp_, f.OutputPoseMessage());
    } else {
        poseFile << f.OutputPoseMessage() << endl;
    }
}

void KeyFrame::ProcessPoseFile() {
    if (poseFile.is_open()) {
        poseFile.close();
    }
    sort(vecTime2Pose.begin(), vecTime2Pose.end(),
         [](const pair<double, string>& kf1, const pair<double, string>& kf2) {
             return kf1.first < kf2.first;
         });
    for (const auto& time2PoseStr : vecTime2Pose) {
        kfPoseFile << time2PoseStr.second << endl;
    }
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
            cvtColor(globalOptFlw.prevImg_, im1, cv::COLOR_GRAY2BGR);
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

size_t KeyFrame::InitializeLandmark(KeyFrame* lastKf) {
    ExtractFeaturetPoints();
    LightglueMatchAndRefineTrackResult(lastKf);

    // 历史关键帧和当前关键帧在当前灰度图上提取到的关键点数量
    return globalOptFlw.GetTrackFeatureNum();
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
    SetBackupPose();
    if (printDiff) {
        const Pose diff = (Tc0w * priorTwc_) * Tcw_;
        cout << "predict pose diff with prior: " << diff << "\n";
    }
}

void KeyFrame::ReleaseMat() {
    grayImg_.release();
    debugGrayImg_.release();
}

// superpoint匹配阈值，使用cos运算量反而更小
#define USE_L2_NORM_DIST 1
constexpr float kWrongMatchNorm = 0.7;
constexpr float kWrongMatchNorm2 = 0.7 * 0.7;

const float kWrongMatchCos = cos(30 * kDeg2Rad);
constexpr float kMNratio = 1.0;  // 有正、反向匹配，不再比较次小

constexpr float kMNratio2 = kMNratio * kMNratio;

constexpr float kMaxObv2EpipolarLineDist = 64.0;  // 640x480

int KeyFrame::SupperPointMatch(KeyFrame* kf,
                               vector<cv::DMatch>& superpointMatches) {
    superpointMatches.reserve(kpts_.rows());

    // Eigen::Matrix3f F12;
    // globalOptFlw.GetFundamentalMatrixF12(cam_, F12);

    Pose T12 = Tcw_ * kf->Twc_;
    Eigen::Matrix3f F12 =
        (cam_->Kinv_[0].transpose() *
         (SkewSymmetric(T12.t_wb_) * T12.q_wb_.toRotationMatrix()) *
         cam_->Kinv_[0])
            .cast<float>();

    auto MatchThread = [&F12](const int startId, const int endId,
                              const Eigen::Matrix<float, Eigen::Dynamic, 2,
                                                  Eigen::RowMajor>& kpts1,
                              const Eigen::Matrix<float, Eigen::Dynamic, 2,
                                                  Eigen::RowMajor>& kpts2,
                              const Eigen::Matrix<float, Eigen::Dynamic, 256,
                                                  Eigen::RowMajor>& desc1s,
                              const Eigen::Matrix<float, Eigen::Dynamic, 256,
                                                  Eigen::RowMajor>& desc2s,
                              vector<cv::DMatch>& matchResult) {
        matchResult.reserve(endId - startId);
        for (int i = startId; i < endId; ++i) {
            Eigen::Vector3f l2 =
                (Eigen::Vector3f(kpts1.row(i)[0], kpts1.row(i)[1], 1.0)
                     .transpose() *
                 F12)
                    .normalized();
            if (abs(l2.x()) < 1e-10 && abs(l2.y()) < 1e-10) {
                continue;
            }

#if USE_L2_NORM_DIST
            float bestScore = 1e6;
#else
            float bestScore = -1e6;
#endif
            int bestMatchIndex2 = -1;
            for (int j = 0; j < desc2s.rows(); ++j) {
                if (ComputeObv2EpipolarLineDist(
                        l2, {kpts2.row(j)[0], kpts2.row(j)[1]}) >
                    kMaxObv2EpipolarLineDist) {
                    continue;
                }

#if USE_L2_NORM_DIST
                // squaredNorm耗时比norm还长
                // float score = (desc1s.row(i) - desc2s.row(j)).squaredNorm();
                float score = (desc1s.row(i) - desc2s.row(j)).norm();
                if (score < bestScore) {
                    bestScore = score;
                    bestMatchIndex2 = j;
                }
            }

            if (bestScore < kWrongMatchNorm) {
                matchResult.emplace_back(i, bestMatchIndex2, bestScore);
            } else {
                matchResult.emplace_back(i, -1, bestScore);
            }
#else
                float score = desc1s.row(i).dot(desc2s.row(j));
                if (score > bestScore) {
                    bestScore = score;
                    bestMatchIndex2 = j;
                }
            }

            if (bestScore > kWrongMatchCos) {
                matchResult.emplace_back(i, bestMatchIndex2, bestScore);
            } else {
                matchResult.emplace_back(i, -1, bestScore);
            }
#endif
        }
    };

    constexpr int kThreadNum = 4;
    const int part = kpts_.rows() / kThreadNum;
    const int remainNum = kpts_.rows() % kThreadNum;
    thread th[kThreadNum];
    vector<vector<cv::DMatch>> matches(kThreadNum);
    for (int i = 0; i < kThreadNum; ++i) {
        const int startId = i * part;
        int endId = startId + part;
        if (i == kThreadNum - 1) {
            endId += remainNum;
        }
        th[i] = thread(MatchThread, startId, endId, ref(kpts_), ref(kf->kpts_),
                       ref(desc_), ref(kf->desc_), ref(matches[i]));
    }
    for (int i = 0; i < kThreadNum; ++i) {
        th[i].join();
    }
    for (int i = 0; i < kThreadNum; ++i) {
        superpointMatches.insert(superpointMatches.end(), matches[i].begin(),
                                 matches[i].end());
    }

    return superpointMatches.size();
}
