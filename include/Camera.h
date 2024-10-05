#ifndef CLASS_CAMERA
#define CLASS_CAMERA

#include "Config.h"

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>


class Camera {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // 实现不考虑畸变参数
    Camera(Config *config);
    // TODO：后续我们都将在归一化平面上进行处理
    Eigen::Vector2d Project2PixelPlane(const Eigen::Vector3d &Pc) const;
    Eigen::Vector3d InverseProject(const Eigen::Vector2i &uv, const double &z = 1.0) const;
    std::vector<Eigen::Vector2d> UndistortPoints(std::vector<cv::Point2i> px) const;
    bool InImagePlaneRange(const Eigen::Vector2d &p) const ;

    double imageScale_; // 需要根据图像缩放fx, fy, cx, cy
    Eigen::Matrix3d K_ = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d K_inv_ = Eigen::Matrix3d::Identity();
    double fx_, fy_, cx_, cy_;
    double k1_, k2_, k3_, k4_;
    double k5_; // 针对TUM数据集的针孔相机
    const double xStart_, yStart_, xStep_, yStep_, xEnd_, yEnd_; // 归一化平面上的显示范围
};

#endif
