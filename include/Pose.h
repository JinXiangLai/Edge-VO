#ifndef CLASS_POSE
#define CLASS_POSE

#include <Eigen/Dense>

class Pose {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Pose(const Eigen::Quaterniond& q_wb, const Eigen::Vector3d& t_wb, const double scale);
    Pose() {}
    Pose(const Pose& T);
    Pose Inverse() const;
    Eigen::Matrix4d ToMatrix4d() const;
    Eigen::Vector3d operator*(const Eigen::Vector3d& p) const;
    Pose operator*(const Pose& T) const;
    friend std::ostream& operator<<(std::ostream& cout, const Pose& T);
    int Size() const;
    void Update(const Eigen::Vector3d& delta_q, const Eigen::Vector3d& delta_t, const double scale);
    
    std::string QwbString() const;
    std::string PwbString() const;

    Eigen::Quaterniond q_wb_ = Eigen::Quaterniond::Identity();
    Eigen::Vector3d t_wb_ = Eigen::Vector3d::Zero();
    double scale_ = 1.0;
};

#endif
