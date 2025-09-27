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
    // EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Landmark(const Eigen::Vector2i& px, KeyFrame* host,
             const std::shared_ptr<Camera> cam, const uint64_t desc,
             const double invZ);
    Landmark() {}
    Eigen::Vector3d GetPcNorm() const;
    Eigen::Vector3d GetPc() const;
    Eigen::Vector3d GetPw() const;
    int Size() const;  // 优化变量的维度
    void Update(const double delta_z, const bool useInvDepth);
    void UpdateUncertainty(const bool updateObv = false);
    bool Converge() const;
    bool ManySupport() const;

    void SetOutOfRange() { outOfRange_ = true; }
    bool IsOutOfRange() const { return outOfRange_; }
    std::vector<Eigen::Vector2d> FindMatches(const KeyFrame& kf2);

    double invZ_ = kInitInvDepth;
    double invDepthCov_ = kInitCov;
    double trueDepth_ = 0.;

    double depthRange_[2] = {0, 0};
    uint64_t descriptor_ = 0;
    // TODO: 结合光度残差分布给定优化的权重值
    int obvTime_ = 0;      // 路标点被看的次数可以反映其可信度
    int failObvTime_ = 0;  // 遮挡或者重复纹理导致失败
    int checkTime_ = 0;
    bool initFromPropagate_ = false;

    //std::shared_ptr<KeyFrame> host_; 需确保host已经由智能指针管理，然后调用shared_from_this()来获取才行，不方便
    KeyFrame* host_;      // cnchor frame
    Eigen::Vector2i uv_;  // host帧下的像素坐标z

    std::map<KeyFrame*, Eigen::Vector2i> target_;
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

    bool noUsed_ = false;

    Eigen::Vector2d matchNextPixel_ = Eigen::Vector2d::Zero();

    bool IsDebugPoint() {
        return config->pixelCount.count(uv_) && trueDepth_ != 0;
    }

    bool ObvUpdate(const double invDepth, const double variance);

    bool FuseInvDepth(const Landmark& lk2);
};

#endif
