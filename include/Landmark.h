#ifndef CLASS_LANDMARK
#define CLASS_LANDMARK

#include <cstdint>
#include <map>
#include <memory>

#include "Camera.h"
#include "KeyFrame.h"

class KeyFrame;

class Landmark {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Landmark(const Eigen::Vector2d &px, KeyFrame *host, const std::shared_ptr<Camera> cam, 
        const uint64_t desc, const double z);
    Landmark() {}
    Eigen::Vector3d GetPcNorm() const;
    Eigen::Vector3d GetPc() const;
    Eigen::Vector3d GetPw() const;
    int Size() const; // 优化变量的维度
    void Update(const double delta_z, const bool useInvDepth);
    void UpdateUncertainty();
    bool Converge() const {
        return uncertainty_ < config->maxDepthConvergeStd && 
            z_ > config->minDepth && z_ < config->maxDepth;
    }
    void SetOutOfRange() {outOfRange_ = true;}
    bool IsOutOfRange() const {return outOfRange_;}
    std::vector<Eigen::Vector2d> FindMatches(const KeyFrame &kf2);

    double z_ = 1.0;
    double depthCov_ = 1e10;
    double invZ_ = 1.0;
    double invDepthCov_ = 1e-10;

    double depthRange_[2] = {0, 0};
    double uncertainty_ = 0;
    uint64_t descriptor_ = 0;
    // TODO: 结合光度残差分布给定优化的权重值
    int obvTime_ = 0; // 路标点被看的次数可以反映其可信度

	//std::shared_ptr<KeyFrame> host_; 需确保host已经由智能指针管理，然后调用shared_from_this()来获取才行，不方便
    KeyFrame *host_; // cnchor frame
    Eigen::Vector2d uv_; // host帧下的像素坐标

	std::map<KeyFrame*, Eigen::Vector2d> target_;
    std::shared_ptr<Camera> cam_;

    // keep FEJ
    std::map<KeyFrame*, Eigen::Matrix<double, 3, 6> > J_Pc2_Twc2; // J_Pc2_Twc2 and J_Pw_Twc1
    std::map<KeyFrame*, Eigen::Matrix3d> J_Pc2_Pw;
    std::vector<Eigen::Matrix<double, 3, 1> > J_Pw_z;
    std::vector<Eigen::Matrix<double, 3, 6> > J_Pw_Twc1;
    void ResetFEJ() {J_Pc2_Twc2.clear(); J_Pc2_Pw.clear(); J_Pw_z.clear(); J_Pw_Twc1.clear();}

    bool outOfRange_ = false; // 多处涉及到同一指针操作，不能直接释放指针
};

#endif
