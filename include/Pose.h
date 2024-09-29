#ifndef CLASS_POSE
#define CLASS_POSE

#include <Eigen/Dense>

class Pose {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Pose(const Eigen::Quaterniond& q_wb, const Eigen::Vector3d& t_wb);
    Pose() {}
    Pose(const Pose& T);
    Pose Inverse() const;
    Eigen::Matrix4d ToMatrix4d() const;
    Eigen::Vector3d operator*(const Eigen::Vector3d &p) const;
    Pose operator*(const Pose& T) const;
    friend std::ostream& operator<<(std::ostream &cout, const Pose& T);
    int Size() const;
    void Update(const Eigen::Vector3d &delta_q, const Eigen::Vector3d &delta_t);

    Eigen::Quaterniond q_wb_ = Eigen::Quaterniond::Identity();
    Eigen::Vector3d t_wb_ = Eigen::Vector3d::Zero();
};

#endif
