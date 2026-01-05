#ifndef CLASS_OPTIMIZER
#define CLASS_OPTIMIZER

#include <fmt/core.h>
#include <memory>
#include <vector>

#include <Eigen/Sparse>

#include "Config.h"
#include "KeyFrame.h"
#include "Landmark.h"
#include "Pose.h"

#define USE_SPARSE_H_MATRIX 1

constexpr int kPoseDim = 6;
constexpr int kPointDim = 1;
constexpr int kMaxNewKFinQueue = 2;
typedef Eigen::Matrix<double, 3, Eigen::Dynamic> DynamicPointMatrix;

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

    bool OptimizeCurFrame(Pose& Twc2, const int curFid, int& totalPointNum,
                          int& usefulPointNum, ResidualInfo& info);

    bool OptimizeCurFrameCeres(Pose& Twc2, const int curFid, int& totalPointNum,
                               int& usefulPointNum, ResidualInfo& info);

    ResidualInfo CalculateResidualCurFrame(
        const std::vector<Eigen::Vector3d>& lk1s,
        const std::vector<Eigen::Vector2d>& obvs, const Pose& Twc2);

    ResidualInfo SetOptimizeLandmarkForTracking(
        const std::vector<std::shared_ptr<Landmark>>& lk1s,
        const std::vector<Eigen::Vector2d>& obvs, const Pose& Twc2,
        const cv::Mat& img, int& canUseNum,
        std::vector<Eigen::Vector3d>& stablePws,
        std::vector<Eigen::Vector2d>& stableObvs);

    void CalculateHandGradiantCurFrame(
        const std::vector<Eigen::Vector3d>& lk1s,
        const std::vector<Eigen::Vector2d>& obvs, const Pose& Twc2,
        Eigen::Matrix<double, kPoseDim, kPoseDim>& H,
        Eigen::Matrix<double, kPoseDim, 1>& g);

    ResidualInfo ResampleStablePwAndObvCurFrame(
        const double sampleRatio, const Pose& Twc2,
        std::vector<Eigen::Vector3d>& stablePws,
        std::vector<Eigen::Vector2d>& stableObvs);

#if USE_SPARSE_H_MATRIX

    Eigen::VectorXd SchurCompleteSolve(const Eigen::SparseMatrix<double>& H,
                                       const Eigen::VectorXd& b,
                                       const int poseNum, const int pointNum,
                                       const bool firstTime,
                                       const bool logOut = false);

    void ConstructSparseMatrixMapTable();

#else
    Eigen::VectorXd SchurCompleteSolve(const Eigen::MatrixXd& H,
                                       const Eigen::VectorXd& b,
                                       const int poseNum, const int pointNum,
                                       const bool& logOut = false);
#endif

    double CalculatePriorCost(const Eigen::VectorXd& deltaX);

    void UpdatePriorDeltaX0(const Eigen::VectorXd& deltaX);

    // TODO： 先不考虑边缘化，而是直接丢弃首帧
    bool MarginalizeOldestKeyFrame();

    void RemoveOldestKeyFrame(const int margKFid);

    bool TransformLandmarkOwnerFromOldestKF(const int margKFid);

    void ShowLocalMap();

    void AddOneKeyFeame(KeyFrame* kf);

    void TriangulateNewLandmark(KeyFrame* kf);

    void ConstructJ_H_b_g(const bool firstTime = false);

    KeyFrame* GetLastKF() { return window_.back(); }

    bool ExecuteWindowOptimize();

    bool ExecuteWindowOptimizeCeres();

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

    void AdaptSetInitLambda(const Eigen::MatrixXd& H, double& lambda);

    void AssignTrackedFeature(
        const std::vector<std::shared_ptr<Landmark>>& lk1s, const Pose& T12,
        const int lvl = 0);

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
                        int& continousNoImprovementNum, double& lambda);

#if USE_SPARSE_H_MATRIX
    double ComputePredictionReduction(const double lambda,
                                      const Eigen::VectorXd& deltaX,
                                      const Eigen::VectorXd& g,
                                      const Eigen::SparseMatrix<double>& H);
#else
    double ComputePredictionReduction(const double lambda,
                                      const Eigen::VectorXd& deltaX,
                                      const Eigen::VectorXd& g,
                                      const Eigen::MatrixXd& H);
#endif

    double ComputePredictionReductionFrame(
        const double lambda, const Eigen::Matrix<double, kPoseDim, 1>& deltaX,
        const Eigen::Matrix<double, kPoseDim, 1>& g,
        const Eigen::Matrix<double, kPoseDim, kPoseDim>& H);

    bool LMstopJudge(const int& continousNoImprovementNum, const double lambda,
                     const ResidualInfo& last, const ResidualInfo& cur,
                     const Eigen::VectorXd& delta);

    void PreSelectLandmarkForTracking(
        std::vector<std::shared_ptr<Landmark>>& lk1s,
        std::vector<Eigen::Vector2d>& obvs);

    int MarkBigResidualLandmarkDelete();

    void CalculateLastKFmeanDepth();

    double GetLastKFmeanDepth() { return lastKFmeanDepth_; }

    void WinBApreAssignMatrixMemory();

    void SetEigenMatrixAll0(Eigen::Matrix<double, -1, -1>& mat,
                            const bool logOut = false);

    void MoveMargKF2FirstPosInWindow(const int margId);

    template <int rows, int cols>
    bool MatrixBlockReset(const int x, const int y, const uint64_t addr) {
        // TODO：尝试使用指针地址
        if (hasResetHessianblock_.count(addr)) {
            return false;
        }
        H_.block<rows, cols>(x, y).setZero();
        hasResetHessianblock_.insert(addr);
        return true;
    }

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
    // double lambda_ = 0.;
    // 普通帧位姿优化使用
    // int maxIte_ = 100;
    bool onlyPoseUpdate_ = false;
    std::shared_ptr<Camera> cam_;
    double lastKFmeanDepth_ = 0.;

   public:
    // edge slam使用
    // 关键帧滑窗优化使用
    std::vector<KeyFrame*> window_;
    std::vector<std::shared_ptr<Landmark>>
        optLandmark_;  // 投影到最新帧能被观测到的才加入，以减小问题规模
#if USE_SPARSE_H_MATRIX
    Eigen::SparseMatrix<double> H_;
    // std::vector<Eigen::Triplet<double>> triplets_;

    //Eigen::SparseMatrix<double> Dinv_;
    // std::vector<Eigen::Triplet<double>> DinvMatTriplets_;
    // 待H_矩阵维度确定且压缩后，构建行索引对应的存储位置，实现O(1)遍历

    // std::vector<std::unordered_map<int, double*>>
    //     colMajorSparseMatrixRowId2DataPtr_;
    std::vector<std::vector<double*>> colMajorSparseMatrixRowId2DataPtr_;
    int sparseHmatrixElementNum_ = 0;
#else
    Eigen::MatrixXd H_;
    // 舒尔补内存，实验发现，大矩阵内存分配比运算耗时！！！
    //Eigen::MatrixXd Dinv_;
#endif

    Eigen::MatrixXd E_, newA_;
    Eigen::MatrixXd J_, Hp_;  // J_的行维度无法提前预知，其涉及的是约束数量
    Eigen::VectorXd g_, g_p_, Dinv_;  // b_，残差的行维度一般是无法提前预知的
    double rpChi2_ =
        0;  // 由边缘化时分解Hp_计算得到，需要计算以避免先验残差为负(事实上，先验残差为负是可接受的，意味着系统往更好的方向优化，因此该常数项不需要考虑)
    Eigen::VectorXd margDeltaX_;  // 边缘化时的状态量增量
    bool margKFstatus_ = false;

    std::map<std::string, std::vector<cv::Mat>> triPointMapDebugImage_;

    std::vector<KeyFrame*> delayEraseKeyframe_;

    cv::VideoWriter debugTrackLostStatusVideoWriter_;
    void WriteDebugTrackLostStatus(const KeyFrame& curF);

    std::unordered_set<uint64_t> hasResetHessianblock_;

    // 分离滑窗优化线程
    bool keepRunWindowBA_ = true;
    KeyFrame* newKF_ = nullptr;
    std::mutex newKFmutex_;
    void RunWindowBA();
    void StopRunBA();
    bool CanAddNewKF() {
        // TODO：隔离三角化与新加KF过程
        return keepRunWindowBA_ && newKFqueue_.size() <= kMaxNewKFinQueue &&
               newKF_ == nullptr;
    }
    double lastWinBAspendTime_ = 1e12;  // ms
    std::queue<KeyFrame*> newKFqueue_;

    // 闭环优化参数
    std::vector<KeyFrame*> vecMargKf_;
    std::mutex vecMargKfMutex_;
    KeyFrame* lastTryLoopNewKf_ = nullptr;  // 避免重复找闭环，由BA线程设置
    std::mutex lastTryLoopNewKfMutex_;
    bool keepRunLoopClosure_ = true;
    std::atomic_bool scaleKFinWindow_ = false;
    std::atomic<int> trackPnPTime = 0;
    int FindLoopClosureKF();
    int FindMatchSuperpoint3Dpos(const KeyFrame* kf1, const KeyFrame* kf2,
                                 DynamicPointMatrix& Pc1,
                                 DynamicPointMatrix& Pc2);
    void RunLoopClosure();
    void StopRunLoopClosure();
    bool Sim3PoseGraphOptimizationCeres2(int fixedIndex, int loopClosureIndex,
                                         const int addKFnumFromWindow,
                                         const Sim3Pose& relativeSim3T12,
                                         std::vector<KeyFrame*>& allKeyframe);
    void UpdateRelativeSim3POSEsT12Ceres2(const DynamicPointMatrix& Pc1,
                                          const DynamicPointMatrix& Pc2,
                                          Sim3Pose& sT12);
    bool SolveCurrentFramePoseAfterLoopClosureCorrectOpencvPnp(Pose& Twc);
    bool GetAndResetScaleKFinWindowFlag();
    std::mutex windowKFposeUpdateMutex;
    // 找到闭环后，应该清空经过闭环优化的帧后再检测新闭环
    // 本质上需要使用词袋来快速找候选帧，然后再使用lightglue确认闭环，因lightglue耗时较长
    // 否则只能检验开头的几帧关键帧
    // 2. 被边缘化的如果是最新帧，那么就不应该加入寻找闭环，因为有很多帧能看到它，且只能在边缘化后的帧找闭环
};

#endif