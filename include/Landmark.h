#ifndef CLASS_LANDMARK
#define CLASS_LANDMARK

#include <cstdint>
#include <map>
#include <memory>

#include "Camera.h"
#include "KeyFrame.h"

class KeyFrame;

constexpr double kInitInvDepth = 5.0;
constexpr double kInitCov = 5.0 * 5.0;

class Landmark {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Landmark(const Eigen::Vector2d& px, KeyFrame* host,
             const std::shared_ptr<Camera> cam, const uint64_t desc,
             const double invZ);
    Landmark() {}
    Eigen::Vector3d GetPcNorm() const;
    Eigen::Vector3d GetPc(const bool useBackUpStatus = false) const;
    Eigen::Vector3d GetPw(const bool useBackUpStatus = false) const;
    int Size() const;  // 优化变量的维度
    void Update(const double delta_z);
    bool Converge() const;
    bool ManySupport() const;

    void SetOutOfRange() { outOfRange_ = true; }
    bool IsOutOfRange() const { return outOfRange_; }

    double invZ_ = kInitInvDepth;
    double invZback_ = invZ_;
    double invDepthCov_ = kInitCov;
    double trueDepth_ = 0.;

    uint64_t descriptor_ = 0;
    // TODO: 结合光度残差分布给定优化的权重值
    int obvTime_ = 0;      // 路标点被看的次数可以反映其可信度
    int failObvTime_ = 0;  // 遮挡或者重复纹理导致失败
    int checkTime_ = 0;
    bool initFromPropagate_ = false;
    bool initialized_ = false;

    //std::shared_ptr<KeyFrame> host_; 需确保host已经由智能指针管理，然后调用shared_from_this()来获取才行，不方便
    KeyFrame* host_;      // cnchor frame
    Eigen::Vector2d uv_;  // host帧下的像素坐标z

    std::unordered_map<KeyFrame*, Eigen::Vector2d> target_;
    std::shared_ptr<Camera> cam_;

    // keep FEJ
    std::map<KeyFrame*, Eigen::Matrix<double, 3, 6>>
        J_Pc2_Twc2;  // J_Pc2_Twc2 and J_Pw_Twc1
    std::map<KeyFrame*, Eigen::Matrix3d> J_Pc2_Pw;
    std::vector<Eigen::Matrix<double, 3, 1>> J_Pw_z;
    std::vector<Eigen::Matrix<double, 3, 6>> J_Pw_Twc1;
    void ResetFEJ();

    mutable bool outOfRange_ =
        false;  // 多处涉及到同一指针操作，不能直接释放指针

    // 跟踪一帧的FEJ
    std::vector<Eigen::Matrix<double, 3, 6>> J_Pc2_T12_;

    Eigen::Vector2d matchNextPixel_ = Eigen::Vector2d::Zero();

    bool IsDebugPoint() {
        return config->pixelCount.count(uv_.cast<int>()) && trueDepth_ != 0;
    }

    bool ObvUpdate(const double invDepth, const double variance);

    void SetTriangulateResult(const double invZ);

    bool FuseInvDepth(const Landmark& lk2);

    bool AbnormalConvergeLandmark();

    // bool CheckInvDepthQualityByProject(const KeyFrame& lastLastFrame);
    bool CheckInvDepthQualitySuccessByProject();

    bool TransformHost2OtherKF(KeyFrame* kf2);

    void BackUpStatus();

    void CopyStatus();

    void SetCanDelete() { canBedelete_ = true; }
    void SetNoUsed() { noUsed_ = true; }
    bool NoUsed() const { return noUsed_; }
    bool CanBeDelete() const { return canBedelete_; }

    //void AddKeyframeTargetObv(KeyFrame* kf, const Eigen::Vector2d& obv);
    //std::mutex mute_;

    // 利用重投影残差检验收敛逆深度的质量
    bool passReprojectCheck_ = false;
    Eigen::Vector2d lastFrameMatchPx_ = Eigen::Vector2d::Zero();
    int continousFailCheckNum_ = 0;
    void AddContinousCheckFailNum() { ++continousFailCheckNum_; }
    int continousPassCheckNum_ = 0;
    void AddContinousCheckPassNum() { ++continousPassCheckNum_; }

   private:
    bool canBedelete_ = false;
    bool noUsed_ = false;
};

#endif
