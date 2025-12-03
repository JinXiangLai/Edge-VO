#ifndef CLASS_KEYFRAME
#define CLASS_KEYFRAME

#include <fmt/core.h>
#include <fstream>
#include <memory>

#include <opencv2/opencv.hpp>

#include "Camera.h"
#include "Landmark.h"
#include "LightGlue.h"
#include "Pose.h"
#include "SuperPoint.h"

#define USE_POINT_MAP_ID 0

#define SUPER_POINT_EXTRACTOR 1

class Landmark;
constexpr double expandRatio = 1.1 * 1.1;

struct TupleHash {
    size_t operator()(const Eigen::Vector2i& v) const {
        const int v1 = v[0];
        const int v2 = v[1];
        return (v1 << 1) + (v2 >> 1);
    }
};

namespace EpipolarMatchType {
constexpr float outOFboundaryORabnormalDepth = -1;
constexpr float repeatTextureORbadDepth = -2;
constexpr float occulsionORnoBestMatch = -3;
constexpr float nanValueNOstereoVisionIssue = -4;
}  // namespace EpipolarMatchType

class KeyFrame {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    KeyFrame(const cv::Mat& img, const Pose& Twc, std::shared_ptr<Camera> cam,
             const int id, const double timestamp, const int level = 1);
    ~KeyFrame();
    KeyFrame() {}
    KeyFrame(const KeyFrame& f);
    void operator=(const KeyFrame& f);

    // 可能需要corase2fine的配准
    size_t InitializeLandmark(KeyFrame* lastKf);
    void SetOutOfRange() { outOfRange_ = true; }
    bool IsOutOfRange() const { return outOfRange_; }
    void Update(const Eigen::Vector3d& delta_q, const Eigen::Vector3d& delta_t);
    void SetTwc(const Pose& Twc, const bool printDiff = false);

    void ReleaseMat();
    void GenerateKeyPoint();
    void SetBackupPose() {
        TcwBack_ = Tcw_;
        TwcBack_ = Twc_;
    }

    int LightglueMatchAndRefineTrackResult(KeyFrame* lastKf);

    unsigned int id_;
    cv::Mat grayImg_, debugGrayImg_;

    Pose Twc_, TwcBack_;
    // TODO： 增加该字段，减小Inverse()次数
    Pose Tcw_, TcwBack_;
    Pose priorTwc_;
    int level_ = 1;
    std::vector<Landmark*>
        landmark_;  // 成员变量内存在指针，需要手写拷贝构造函数

    bool outOfRange_ = false;
    int convergeEdgeNum_ = 0;
    int updateFrameCount_ = 0;

    // debug 优化算法
    cv::Mat depthImage_;

    void CopyStatus();

    void BackUpStatus();

    std::map<std::string, int> matchResultStatiscs_;
    void ReportMatchResult();
    void AddReportElement(const std::string& key);
    void ResetDebugMessage();
    std::string OutputPoseMessage() const;

    void OpticalFlowTrackExecute(const cv::Mat& prevImg, const cv::Mat& curImg);

    std::ofstream invDepthUncertaintyFile_;
    bool firstWriteUncertainty_ = true;

    void OpticalFlowTrackLandmark(const KeyFrame& f2);
    struct OpticalFlowStruct {
        cv::Mat prevImg_;
        std::vector<cv::Point2f> prevPts_;
        std::vector<Landmark*> trackLandmark_;
        size_t totalFeatureCreated_ = 0;
        size_t historyLandmarkNum_ = 0;  // 记录历史跟踪点的数量以区分上一KF的
        double meanParallax_ = 0.;
        int usefulParallaxNum_ = 0;
        size_t GetTrackFeatureNum() { return prevPts_.size(); }
        double GetTrackFeatureRatio() {
            return double(GetTrackFeatureNum()) / totalFeatureCreated_;
        }
        size_t SetTotalFeatureCreated() {
            totalFeatureCreated_ = GetTrackFeatureNum();
            return totalFeatureCreated_;
        }
        void Reset() {
            prevImg_.release();
            prevPts_.clear();
            trackLandmark_.clear();
            totalFeatureCreated_ = 0;
            historyLandmarkNum_ = 0;
        }
        void Set(const cv::Mat& img, const std::vector<cv::Point2f>& prevPts,
                 const std::vector<Landmark*>& landmark,
                 const int totalFeatureCreated, const int historyLandmarkNum);
    };
    void SetOpticalFlowStructCurFrame();
    double TrackWithOpticalFlow(const KeyFrame& kf2, int& findMatchNum);
    void ExtractFastPoints();
    void GenerateUndistordMap();
    int RemoveNoInitializeLongFeature();
    bool ExtractFastPointEachGrid(const int diffRow, const int diffCol,
                                  const int fastTh1, cv::Point2f& fast);
    void ExtractFastPointEachImage();

    Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor> kpts_;
    Eigen::Matrix<float, Eigen::Dynamic, 256, Eigen::RowMajor> desc_;
    Eigen::Vector2d GetObv(const int kpRow) const {
        return {kpts_(kpRow, 0), kpts_(kpRow, 1)};
    }

    static OpticalFlowStruct optFlw;

#if defined(WRITE_MATCH_PAIR_IMAGE)
    void DrawBestMatchEachFrame(const Eigen::Vector2i& kp1,
                                const Eigen::Vector2i& matchKp2,
                                const cv::Mat& debugImg2,
                                const bool drawOpticalFlow = false);
    void DrawEpipolarMatchEachFrame(const Eigen::Vector2i& kp1,
                                    const Eigen::Vector2i& lp2Start,
                                    const Eigen::Vector2i& lp2End,
                                    const Eigen::Vector2i& matchKp2,
                                    const cv::Mat& debugImg2);
    void DrawFailEpipolarMatchEachFrame(const Eigen::Vector2i& kp1,
                                        const Eigen::Vector2i& lp2Start,
                                        const Eigen::Vector2i& lp2End,
                                        const cv::Mat& debugImg2);
    void WriteDebugImage2VideoEachFrame(const int kf2Id,
                                        const std::string& debugVideoName);
    void DrawTriangulateCase(
        const double estD1, const double estD2, const Landmark& lk1,
        const Eigen::Vector2i& epipolarP1, const Eigen::Vector2i& matchKp2,
        const Eigen::Vector2i& farPx2, const Eigen::Vector2i& nearPx2,
        const cv::Mat& debugImg2, const Pose& T12, const bool success = false);
    void WriteDebugTriangulateCase2Video();
    cv::Mat videoBestMatchDebugImg_;
    cv::Mat videoEpipolarMatchDebugImg_;
    cv::Mat videoEpipolarFailMatchDebugImg_;
    static cv::VideoWriter debugVideoWriter;
    static cv::VideoWriter debugTriangulateWriter;
    std::map<std::string, std::vector<cv::Mat>> triPointMapDebugImage_;
    // TODO：保留一些Landmark的深度收敛过程
#endif

    static cv::Mat map1, map2;
    static Pose Tc0w;  // 运行时世界系c0，到数据集真值轨迹w的变换
    static std::mutex
        mutexForSyncView3Dstatus;  // 由于地图点在关键帧失效时会被删除，因此需要同步
    static std::unordered_set<KeyFrame*> kfOn3Dshow;
    static std::shared_ptr<Camera> cam_;
    static cv::Size eachGridSize;
    static cv::Ptr<cv::FastFeatureDetector> detectorTh1, detectorTh2;
    void CalculateEachGridForExtractFast();

    void InitFastDetector();

    double timestamp_ = 0.;

    // 保留位姿信息
    static std::ofstream poseFile;
    static std::ofstream kfPoseFile;
    static std::vector<std::pair<double, std::string>> vecTime2Pose;
    static void InitPoseFileMessage();
    static void WritePoseMessage2File(const KeyFrame& f);
    static void ProcessPoseFile();

    // superpoint和lightglue
    static std::shared_ptr<SuperPoint> superpointPtr;
    static std::shared_ptr<LightGlue> lightgluePtr;
};

#endif
