#ifndef CLASS_KEYFRAME
#define CLASS_KEYFRAME

#include <fstream>
#include <memory>

#include <opencv2/opencv.hpp>

#include "Camera.h"
#include "Landmark.h"
#include "Pose.h"

#define USE_POINT_MAP_ID 0

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
    // EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    KeyFrame(const cv::Mat& img, const Pose& Twc, std::shared_ptr<Camera> cam,
             const int id, const int level = 1);
    ~KeyFrame();
    KeyFrame() {}
    KeyFrame(const KeyFrame& f);
    void operator=(const KeyFrame& f);

    // 可能需要corase2fine的配准
    size_t InitializeLandmark(const KeyFrame* lastKf);
    void SetOutOfRange() { outOfRange_ = true; }
    bool IsOutOfRange() const { return outOfRange_; }
    void Update(const Eigen::Vector3d& delta_q, const Eigen::Vector3d& delta_t);
    void SetTwc(const Pose& Twc);

    void ReleaseMat();
    void GenerateKeyPoint();

    unsigned int id_;
    cv::Mat grayImg_, debugGrayImg_;

    std::shared_ptr<Camera> cam_;
    Pose Twc_, TwcBack_;
    // TODO： 增加该字段，减小Inverse()次数
    Pose Tcw_, TcwBack_;
    Pose priorTwc_;
    int level_ = 1;
    std::vector<Eigen::Vector2d> unKeypoints_;  // 像素平面上的去畸变点
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

    void OpticalFlowTrackExcute(const cv::Mat& prevImg, const cv::Mat& curImg,
                                std::vector<cv::Point2f>& prevPts,
                                std::vector<Landmark*>& prevTrackLandmark);

    std::ofstream invDepthUncertaintyFile_;
    bool firstWriteUncertainty_ = true;

    void OpticalFlowTrackLandmark(const KeyFrame& f2);
    struct OpticalFlowStruct {
        cv::Mat prevImg_;
        std::vector<cv::Point2f> prevPts_;
        std::vector<Landmark*> trackLandmark_;
        // TODO： 保留历史关键帧的跟踪结果
        std::vector<cv::Point2f> prevHistoryPts_;
        std::vector<Landmark*> trackHistoryLandmark_;
        int totalFeatureCreated_ = 0;
        int GetTrackFeatureNum() {
            return prevPts_.size() + prevHistoryPts_.size();
        }
        double GetTrackFeatureRatio() {
            return double(GetTrackFeatureNum()) / totalFeatureCreated_;
        }
        int SetTotalFeatureCreated() {
            totalFeatureCreated_ = GetTrackFeatureNum();
            return totalFeatureCreated_;
        }
    };
    void SetOpticalFlowStructCurFrame();
    double TrackWithOpticalFlow(const KeyFrame& kf2, int& findMatchNum);
    void ExtractFastPoints(const OpticalFlowStruct& lastKFoptFlw);
    void DrawOpticalMatchImg();
    void GenerateUndistordMap();

    OpticalFlowStruct optFlw_;

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
    void WriteDebugImage2VideoEachFrame(const int kf2Id, const std::string& debugVideoName);
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
};

#endif
