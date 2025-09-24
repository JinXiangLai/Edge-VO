#ifndef CLASS_KEYFRAME
#define CLASS_KEYFRAME

#include <fstream>
#include <memory>

#include "Camera.h"
#include "Landmark.h"
#include "Pose.h"

#define USE_POINT_MAP_ID 0

class Landmark;

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
    void CannyEdgeDetect();
    void ExtractEdge();
    void GenerateDTandDerivative();
    size_t GenerateLandmark(
        KeyFrame& kf1, std::vector<std::vector<Eigen::Vector2d> >& debugGoodKp1,
        std::vector<std::vector<Eigen::Vector2d> >& debugGoodKp2,
        const int equalparts);
    size_t InitializeLandmark();
    int ReuseLandmark(KeyFrame* kf1);
    double UpdateDepth(const KeyFrame& kf2);
    void SetOutOfRange() { outOfRange_ = true; }
    bool IsOutOfRange() const { return outOfRange_; }
    void Update(const Eigen::Vector3d& delta_q, const Eigen::Vector3d& delta_t);
    void SetTwc(const Pose& Twc);
    int TrackLandmarkByEpilorLine(const KeyFrame& kf1);
    double CullingBadDepth(KeyFrame* kf2);
    double CalculateSSD(double* v1, double* v2, double avg1, double avg2,
                        const int desLen);
    std::vector<double> CalculateDescriptor(const cv::Mat& grayImg,
                                            const Eigen::Vector2d& px,
                                            const Eigen::Vector2d& epNorm,
                                            const int len = 5);
    void ReleaseMat();
    void FuseDepth();
    bool MoveNearPx2IntoBoundary(Eigen::Vector2d& pClose,
                                 const Eigen::Vector2d& ep2,
                                 const Eigen::Vector2d& pFar);

    // 输出far2->near2，以及相机1的极点，绘制p1->epipolarP1
    double FindMatchesWithEpipolarConstraintOnImagePlane(
        const KeyFrame* kf2, Landmark* lk1, Eigen::Vector2d& bestPx2,
        Eigen::Vector2d& farPx, Eigen::Vector2d& nearPx,
        Eigen::Vector2d& epipolarP1, const bool drawMatch = true);

    void GenerateKeyPoint();

    unsigned int id_;
    cv::Mat grayImg_, debugGrayImg_;
    // canny边缘图像已经去畸变了
    std::vector<cv::Mat> edgeImg_, dist_, dx_, dy_;

    std::shared_ptr<Camera> cam_;
    Pose Twc_;
    // TODO： 增加该字段，减小Inverse()次数
    Pose Tcw_;
    Pose priorTwc_;
    int level_ = 1;
    std::vector<std::vector<Eigen::Vector2i> > unPx_;  // 像素平面上的去畸变点
    std::vector<Landmark*>
        landmark_;  // 成员变量内存在指针，需要手写拷贝构造函数
    // std::vector<Eigen::Matrix<float, kDescriptorPatchSize, 1> > descriptor_;
    std::vector<uint64_t> descriptor_;
    static constexpr int descDim = 63;

#if USE_POINT_MAP_ID
    std::unordered_map<Eigen::Vector2i, int, TupleHash>
        pointMapId_;  // 像素坐标与vector索引的映射
#else
    std::unordered_map<int, int> pointMapId_;  // key: y*width + x
#endif

    bool outOfRange_ = false;
    int convergeEdgeNum_ = 0;
    int updateFrameCount_ = 0;

    // debug 优化算法
    cv::Mat depthImage_;

    std::map<std::string, int> matchResultStatiscs_;
    void ReportMatchResult();
    void AddReportElement(const std::string& key);
    void ResetDebugMessage();

    std::ofstream invDepthUncertaintyFile_;
    bool firstWriteUncertainty_ = true;

#if defined(WRITE_MATCH_PAIR_IMAGE)
    void DrawBestMatchEachFrame(const Eigen::Vector2i& kp1,
                                const Eigen::Vector2i& matchKp2,
                                const cv::Mat& debugImg2);
    void DrawEpipolarMatchEachFrame(const Eigen::Vector2i& kp1,
                                    const Eigen::Vector2i& lp2Start,
                                    const Eigen::Vector2i& lp2End,
                                    const Eigen::Vector2i& matchKp2,
                                    const cv::Mat& debugImg2);
    void DrawFailEpipolarMatchEachFrame(const Eigen::Vector2i& kp1,
                                        const Eigen::Vector2i& lp2Start,
                                        const Eigen::Vector2i& lp2End,
                                        const cv::Mat& debugImg2);
    void WriteDebugImage2VideoEachFrame(const int kf2Id);
    void DrawTriangulateCase(
        const double estD1, const double estD2, const Eigen::Vector2i& kp1,
        const Eigen::Vector2i& epipolarP1, const Eigen::Vector2i& matchKp2,
        const Eigen::Vector2i& farPx2, const Eigen::Vector2i& nearPx2,
        const cv::Mat& debugImg2, const bool success = false);
    void WriteDebugTriangulateCase2Video(const std::string& caseName,
                                         const Pose& T12, cv::Mat& img);
    cv::Mat videoBestMatchDebugImg_;
    cv::Mat videoEpipolarMatchDebugImg_;
    cv::Mat videoEpipolarFailMatchDebugImg_;
    cv::Mat videoFailTriangulateDebugImg_;
    cv::Mat videoSuccessTriangulateDebugImg_;
    static cv::VideoWriter debugVideoWriter;
    static cv::VideoWriter debugTriangulateWriter;
    // TODO：保留一些Landmark的深度收敛过程
#endif
};

#endif
