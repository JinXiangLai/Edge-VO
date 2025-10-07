#include "Camera.h"

using namespace std;
using namespace cv;

Camera::Camera(Config* config)
    : imageScale_(config->imageScale),
      fx_(config->intrinstic[0] * config->imageScale),
      fy_(config->intrinstic[2] * config->imageScale),
      cx_(config->intrinstic[1] * config->imageScale),
      cy_(config->intrinstic[3] * config->imageScale),
      k1_(config->distortion[0]),
      k2_(config->distortion[1]),
      k3_(config->distortion[2]),
      k4_(config->distortion[3]),
      xStep_(1.0 / fx_),
      yStep_(1.0 / fy_),
      xStart_(-cx_ / fx_),
      yStart_(-cy_ / fy_),
      xEnd_(-xStart_),
      yEnd_(-yStart_) {
    const int level = config->pyrLevel;
    K_.resize(level);
    Kinv_.resize(level);
    for (int i = 0; i < level; ++i) {
        const double ratio = 1.0 / pow(2, i);
        K_[i] << fx_ * ratio, 0, cx_ * ratio, 0, fy_ * ratio, cy_ * ratio, 0, 0,
            1;
        Kinv_[i] = K_[i].inverse();
    }
    if (config->model == "pinhole") {
        k5_ = config->distortion[4];
    }
}

Eigen::Vector2d Camera::Project2PixelPlane(const Eigen::Vector3d& Pc,
                                           const int level) const {
    const Eigen::Vector3d p_norm = Pc / Pc.z();
    Eigen::Vector2d res{K_[level].row(0)[0] * p_norm[0] + K_[level].row(0)[2],
                        K_[level].row(1)[1] * p_norm[1] + K_[level].row(1)[2]};
    return res;
}

// Pc投影到归一化平面后，需要进行畸变再作用到K得到像素坐标
// 像素坐标经过K投影到归一化平面后，需要去畸变
// OK，我们将所有的一切都转到归一化平面上并去畸变，
// 然后就可以生成一个新的无畸变的边缘图像了
// 归一化平面上，X轴分辨率为1/fx米、Y轴分辨率为1/fy米
vector<Eigen::Vector2i> Camera::UndistortPoints(vector<Eigen::Vector2i> px,
                                                const int level) const {
    Mat D;
    if (config->model == "fisheye") {
        D = (cv::Mat_<float>(4, 1) << k1_, k2_, k3_, k4_);
    } else if (config->model == "pinhole") {
        D = (cv::Mat_<float>(5, 1) << k1_, k2_, k3_, k4_, k5_);
    }
    Mat R = cv::Mat::eye(3, 3, CV_32F);
    const double ratio = 1 / pow(2, level);
    Mat K = (cv::Mat_<float>(3, 3) << fx_ * ratio, 0, cx_ * ratio, 0,
             fy_ * ratio, cy_ * ratio, 0, 0, 1);

    vector<Point2f> pxs;
    for (const Eigen::Vector2i& p : px) {
        pxs.push_back({static_cast<float>(p.x()), static_cast<float>(p.y())});
    }
    // 函数的输入、输出均是像素平面上的点
    // 经过显示去畸变前后图像检验，去畸变函数是正确且有效的
    if (config->model == "fisheye") {
        cv::fisheye::undistortPoints(pxs, pxs, K, D, R, K);
    } else if (config->model == "pinhole") {
        cv::undistortPoints(pxs, pxs, K, D, R, K);
    }
    vector<Eigen::Vector2i> res;
    for (const Point2f& p : pxs) {
        res.push_back({int(p.x), int(p.y)});
    }

    return res;
}

//template <typename T>
//Eigen::Vector3d Camera::InverseProject(const Eigen::Matrix<T, 2, 1>& uv,
//                                       const double& z, const int level) const {
//    Eigen::Vector3d p(double(uv[0]), double(uv[1]), 1.0);
//    // {(x-cx)/fx, (y-cy)/fy}
//    p = Kinv_[level] * p;
//    // cout << "K.inv * p: " << p.transpose() << endl;
//    return p * z;
//}

// 暂时无用
bool Camera::InImagePlaneRange(const Eigen::Vector2d& p,
                               const int level) const {
    return p.x() > (xStart_ + xStep_) && p.x() < (xEnd_ - xStep_) &&
           p.y() > (yStart_ + yStep_) && p.y() < (yEnd_ - yStep_);
}
