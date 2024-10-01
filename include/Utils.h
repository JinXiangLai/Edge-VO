#ifndef UTILS
#define UTILS

#include <cstdint>
#include <iostream>
#include <memory>
#include <set>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>
#include <Eigen/Dense>

#include "Config.h"
#include "Pose.h"
#include "WheelCameraCalib.h"
#include "Camera.h"
#include "Landmark.h"
#include "KeyFrame.h"

class KeyFrame;
class Landmark;

// 距离变换是计算前景到背景的距离

cv::Mat GetDistanceTransform(cv::Mat img);

std::vector<Eigen::Vector3d> TransformPoint2Pc(const Pose &T, std::vector<Eigen::Vector3d> &ps);

void CaculateDerivative(const cv::Mat &dist, cv::Mat &dx, cv::Mat &dy);

bool InRange(const cv::Mat &img, const Eigen::Vector2i &p);

Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d &v);

double BilinearInterpolate(const cv::Mat &img, const Eigen::Vector2d &p);

void Assert(bool a, const std::string &s);

void ShowImage(const cv::Mat &img, const std::string &name, const bool show = true);

Eigen::Vector3d Quat2RPY(const Eigen::Quaterniond &_q);

std::ostream& operator<<(std::ostream &cout, const Pose& T);

cv::Mat DrawMatch(const cv::Mat &img1, const cv::Mat &img2, const std::vector<Eigen::Vector2d> &kp1, 
    const std::vector<Eigen::Vector2d> &kp2, const std::string &name = "matches", 
    const int ratio = 1, const int jump = 10);
    
int DrawMatch(KeyFrame *kf1, KeyFrame *kf2, const std::string &name="Last track first kf matches");

int DrawMatch(std::vector<Landmark*> &ps, KeyFrame *kf2, const std::string &name="Project landmark to last frame");

std::vector<Eigen::Vector2d> FindMatches(const Landmark &pc1, const KeyFrame &kf2, const Pose &T21, const Camera &cam);

std::vector<Eigen::Vector2d> FindMatchesWithEpipolarConstraintOnImagePlane(const Eigen::Vector2d &kp1, const cv::Mat &edgeImg, 
    const Pose &T21, const Camera &cam);

Eigen::Vector3d Triangulate(const Eigen::Vector2d &kp2, const Pose &T21, const Camera &cam);

Eigen::Vector3d Triangulate(const Eigen::Vector2d &kp1, const Eigen::Vector2d &kp2, const Pose &T21, const Camera &cam);

bool UpdateLandmarkDepth(const std::vector<Eigen::Vector2d> &kp2, const Pose &T21, const Camera &cam, Landmark &landmark);

Pose ConvertRPYandPostion2Pose(const Eigen::Vector3d &rpy, const Eigen::Vector3d &t, const double deg2rad = kDeg2Rad);

void varifyTriangulate();

std::vector<Eigen::Vector2d> CannyEdgeDetect(const cv::Mat &img, cv::Mat &edgeImg, const Camera &cam);

size_t LoadImages(const std::string& strDirectory, std::vector<std::string>& vstrImages, std::vector<double>& vTimeStamps);

size_t LoadPriorOdom(const std::string &strDirectory, std::vector<Eigen::Matrix<double, 8, 1>> &vPriorPose);

void FindImageAndPose(const int idx, const std::vector<std::string> &vstrImages, const std::vector<double> vTimeStamps, 
    const std::vector<Eigen::Matrix<double, 8, 1>> vPriorPose, const WheelCameraCalib &calib, std::vector<cv::Mat> &imgs, 
    std::vector<Pose> &vTwc, const int needNum = 2);

void GetImageAndPose(const int idx, const std::vector<std::string> &vstrImages, const std::vector<double> vTimeStamps, 
    const std::vector<Eigen::Matrix<double, 8, 1>> vPriorPose, const WheelCameraCalib &calib, cv::Mat &img, 
    Pose &Twc);

// double CalculateScore(const Eigen::Matrix<float, kDescriptorPatchSize, 1> &d1, const Eigen::Matrix<float, kDescriptorPatchSize, 1> &d2);

uint64_t CalculateDescriptor(const cv::Mat &grayImg, const Eigen::Vector2i &px);

int CalculateDescriptorScore(const uint64_t v1, const uint64_t v2);

void GetProjectRange(const Landmark &lp, const Pose& T21, const Camera &cam, Eigen::Vector2i &xRange, 
    Eigen::Vector2i &yRange);

void ShowPointCloud(const std::vector<Landmark*> &ps);

void ShowPointCloud(const std::vector<Landmark*> &ps1, const std::vector<Landmark*> &ps2, 
    const std::string &windowName = "Point cloud", const double zOffset = 0.0);

double GetOnePixelUncertainty(const Eigen::Vector3d &t12, const Eigen::Vector3d &pc1, const double f);

bool NeedNewKF(const KeyFrame *kf, const KeyFrame *f);

void ShowPointCloud(const std::set<Landmark* > &ps);

bool IsFastPoint(const cv::Mat &gray, const Eigen::Vector2i px);
#endif