#ifndef CLASS_OPTIMIZER
#define CLASS_OPTIMIZER

#include <memory>
#include <vector>

#include "Config.h"
#include "KeyFrame.h"
#include "Landmark.h"
#include "Pose.h"

class Optimizer {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Optimizer(std::shared_ptr<Camera> cam, const double lambda = 1.0,
              const int maxIte = 100, const bool onlyPoseUpdate = false);

    ~Optimizer();

    struct ResidualInfo {
        double cost = 0;
        int totalConstraintNum = 0;
        int usefulLandmarkNum = 0;
        double meanCost = 0;
        double priorConstraintChi2 = 0.;
    };

    bool OptimizeCurFrame(KeyFrame::OpticalFlowStruct& optFlw, Pose& Twc2,
                          const int curFid, int& totalPointNum,
                          int& usefulPointNum, ResidualInfo& info);

    ResidualInfo CalculateResidualCurFrame(
        const std::vector<Landmark*>& lk1s,
        const std::vector<Eigen::Vector2d>& obvs, const Pose& Twc2);

    ResidualInfo SetOptimizeLandmarkForTracking(
        const std::vector<Landmark*>& lk1s,
        const std::vector<Eigen::Vector2d>& obvs, const Pose& Twc2,
        const cv::Mat& img, int& canUseNum, std::vector<Landmark*>& stableLks,
        std::vector<Eigen::Vector2d>& stableObvs);

    void CalculateHandGradiantCurFrame(const std::vector<Landmark*>& lk1s,
                                       const std::vector<Eigen::Vector2d>& obvs,
                                       const Pose& Twc2, Eigen::MatrixXd& H,
                                       Eigen::VectorXd& g);

    Eigen::VectorXd SchurCompleteSolve(const Eigen::MatrixXd& H,
                                       const Eigen::VectorXd& b,
                                       const int poseNum, const int pointNum,
                                       const int poseDim = 6,
                                       const int pointDim = 1,
                                       const bool& logOut = false);

    double CalculatePriorCost(const Eigen::VectorXd& deltaX);

    void UpdatePriorDeltaX0(const Eigen::VectorXd& deltaX);

    // TODO： 先不考虑边缘化，而是直接丢弃首帧
    bool MarginalizeOldestKeyFrame();

    void RemoveOldestKeyFrame(const int margKFid);

    bool TransformLandmarkOwnerFromOldestKF(const int margKFid);

    void ShowLocalMap();

    void AddOneKeyFeame(KeyFrame* kf);

    void ConstructJ_H_b_g(const bool logOut = false);

    KeyFrame* GetLastKF() { return window_.back(); }

    bool ExecuteWindowOptimize();

    ResidualInfo CalculateResidualWindow(const bool useBackUpStatus = false);

    ResidualInfo SetOptimizeStatusVariableForWindowBA(
        KeyFrame* const margKF = nullptr);

    void DebugOptlandmarkStatus(const size_t num = 20,
                                const std::string& name = "debug");

    bool SlidingWindowOptimize(KeyFrame* curKF);

    // rho[0]经胡伯核的损失函数值，rho[1]胡伯核关于chi2的一阶导数
    // 注意：胡伯核函数只能处理标量
    void HuberLoss(const double chi2, Eigen::Vector2d& rho);

    int SelectOneKF2Marginalization(const KeyFrame& curKF);

    int SampleUsefulLandmark(const int margKFid);

    void ConstructRelativePoseConstraint(Eigen::MatrixXd& H,
                                         Eigen::VectorXd& g);

    bool TrackLocalMap(KeyFrame* kf2, bool& needNewKFbySight);

    void SetInitLambda(const double lambda) { lambda_ = lambda; }

    void AdaptSetInitLambda();

    void CullingErrorLandmark(KeyFrame* curF = nullptr);

    void AssignTrackedFeature(const std::vector<Landmark*>& lk1s,
                              const Pose& T12, const int lvl = 0);

    void RemoveOneKeyframe(const KeyFrame& curF);

    void DrawTriangulateCase(const double estD1, const Landmark& lk1,
                             const Eigen::Vector2i& matchKp2,
                             const cv::Mat& debugImg2, const Pose& T12,
                             const bool success = false);

    void DrawProjectCase(const Landmark& lk1, const Eigen::Vector2d& matchKp2,
                         const cv::Mat& debugImg2, const Pose& Twc2);

    void WriteDebugTriangulateCase2Video(const int curFid);

    void UpdateStatusVariables(const Eigen::VectorXd& deltaX,
                               const size_t startPoseId = 0);

    bool CalculatePriorCostChi2(const Eigen::VectorXd& deltaX);

    void UpdateLMlambda(const ResidualInfo& lastCost,
                        const ResidualInfo& newCost,
                        const double predictReduction, bool& accept,
                        int& continousNoImprovementNum,
                        double& costRelativeAbsDiff);

    double ComputePredictionReduction(const Eigen::VectorXd& deltaX,
                                      const Eigen::VectorXd& g,
                                      const Eigen::MatrixXd& H);

    bool LMstopJudge(const int& continousNoImprovementNum,
                     const double& costRelativeAbsDiff,
                     const Eigen::VectorXd& delta);

    void PreSelectLandmarkForTracking(KeyFrame::OpticalFlowStruct& optFlw,
                                      std::vector<Landmark*>& lk1s,
                                      std::vector<Eigen::Vector2d>& obvs);

    int MarkBigResidualLandmarkDelete();

    void CalculateLastKFmeanDepth();

    double GetLastKFmeanDepth() { return lastKFmeanDepth_; }

    void Reset() {
        window_.clear();
        optLandmark_.clear();
        triPointMapDebugImage_.clear();
        delayEraseKeyframe_.clear();
    }

    cv::VideoWriter debugTriangulateWriter_;

   private:
    // 等价于在成本函数中增加了 0.5*λ*ΔX'*ΔX这一正则项，
    // 因此，λ越大，ΔX须越小
    double lambda_ = 0.;
    // 普通帧位姿优化使用
    int maxIte_ = 100;
    bool onlyPoseUpdate_ = false;
    std::shared_ptr<Camera> cam_;
    double lastKFmeanDepth_ = 0.;

   public:
    // edge slam使用
    // 关键帧滑窗优化使用
    std::vector<KeyFrame*> window_;
    std::vector<Landmark*>
        optLandmark_;  // 投影到最新帧能被观测到的才加入，以减小问题规模
    Eigen::MatrixXd J_, H_, Hp_;  // J_的行维度无法提前预知，其涉及的是约束数量
    Eigen::VectorXd g_, g_p_;  // b_，残差的行维度一般是无法提前预知的
    double rpChi2_ =
        0;  // 由边缘化时分解Hp_计算得到，需要计算以避免先验残差为负(事实上，先验残差为负是可接受的，意味着系统往更好的方向优化，因此该常数项不需要考虑)
    Eigen::VectorXd margDeltaX_;  // 边缘化时的状态量增量
    bool margKFstatus_ = false;

    std::map<std::string, std::vector<cv::Mat>> triPointMapDebugImage_;

    std::vector<KeyFrame*> delayEraseKeyframe_;
};

#endif