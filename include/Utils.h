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

#define USE_SSD

class KeyFrame;
class Landmark;

// 距离变换是计算前景到背景的距离

cv::Mat GetDistanceTransform(cv::Mat img);

std::vector<Eigen::Vector3d> TransformPoint2Pc(const Pose &T, std::vector<Eigen::Vector3d> &ps);

void CaculateDerivative(const cv::Mat &dist, cv::Mat &dx, cv::Mat &dy);

bool InRange(const cv::Mat &img, const Eigen::Vector2i &p);

Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d &v);

template<typename T>
    double BilinearInterpolate(const cv::Mat &img, const Eigen::Vector2d &p);

template<typename T>
    double BilinearInterpolate(const cv::Mat &img, const Eigen::Vector2d &p) {
        if(!InRange(img, p.cast<int>())) {
            return 0;
        }
        
        const int col = img.cols;
        const int row = img.rows;
        const int x = int(p.x());
        const int y = int(p.y());
        if(x == col-1 || x == 0 || y == row-1 || y == 0) {
            return img.at<T>(y, x);
        }

        /****** 双线性插值 ******
        * +---+---+
        * + v1+ v2+
        * +---+---+
        * + v3+ v4+
        * +---+---+
        ***********************/
        float v1 = img.at<T>(y, x);
        float v2 = img.at<T>(y, x+1);
        float v3 = img.at<T>(y+1, x);
        float v4 = img.at<T>(y+1, x+1);
        const double wx = p.x() - x;
        const double wy = p.y() - y;
        const double w1 = (1-wx) * (1-wy);
        const double w2 = wx * (1-wy);
        const double w3 = (1-wx) * wy;
        const double w4 = wx * wy;
        // cout << "w1+w2+w3+w4: " << (w1+w2+w3+w4) << endl; // equal to 1
        return w1*v1 + w2*v2 + w3*v3 + w4*v4;
    }

void Assert(bool a, const std::string &s);

void ShowImage(const cv::Mat &img, const std::string &name, const bool show = true);

Eigen::Vector3d Quat2RPY(const Eigen::Quaterniond &_q);

std::ostream& operator<<(std::ostream &cout, const Pose& T);

cv::Mat DrawMatch(const cv::Mat &img1, const cv::Mat &img2, const std::vector<Eigen::Vector2d> &kp1, 
    const std::vector<Eigen::Vector2d> &kp2, const std::string &name = "matches", 
    const int ratio = 1, const int jump = 10);
    
int DrawMatch(KeyFrame *kf1, KeyFrame *kf2, const std::string &name="Last track first kf matches");

int DrawMatch(std::vector<Landmark*> &ps, KeyFrame *kf2, const std::string &name="Project landmark to last frame");

std::vector<Eigen::Vector2d> FindMatches(const Landmark &lk1, const KeyFrame &kf2, const Pose &T21, const Camera &cam);

std::vector<Eigen::Vector2d> FindMatchesAlongEpipolar(const Landmark &lk1, const KeyFrame &kf2, const Pose &T21, const Camera &cam);

std::vector<Eigen::Vector2d> FindMatchesWithEpipolarConstraintOnImagePlane(const Eigen::Vector2d &kp1, const cv::Mat &edgeImg, 
    const Pose &T21, const Camera &cam);

Eigen::Vector3d Triangulate(const Eigen::Vector2d &kp2, const Pose &T21, const Camera &cam);

Eigen::Vector3d Triangulate(const Eigen::Vector2d &kp1, const Eigen::Vector2d &kp2, const Pose &T21, const Camera &cam);

bool UpdateLandmarkDepth(const std::vector<Eigen::Vector2d> &kp2, const Pose &T21, const Camera &cam, Landmark &lk);

Pose ConvertRPYandPostion2Pose(const Eigen::Vector3d &rpy, const Eigen::Vector3d &t, const double deg2rad = kDeg2Rad);

void varifyTriangulate();

std::vector<Eigen::Vector2d> CannyEdgeDetect(const cv::Mat &img, cv::Mat &edgeImg, const Camera &cam);

size_t LoadImages(const std::string& strDirectory, std::vector<std::string>& vstrImages, 
    std::vector<double>& vTimeStamps, const std::string &imgSuffix=".jpg");

size_t LoadPriorOdom(const std::string &strDirectory, std::vector<Eigen::Matrix<double, 8, 1>> &vPriorPose);

void FindImageAndPose(const int idx, const std::vector<std::string> &vstrImages, const std::vector<double> vTimeStamps, 
    const std::vector<Eigen::Matrix<double, 8, 1>> vPriorPose, const WheelCameraCalib &calib, std::vector<cv::Mat> &imgs, 
    std::vector<Pose> &vTwc, const int needNum = 2);

void GetImageAndPose(const int idx, const std::vector<std::string> &vstrImages, const std::vector<double> vTimeStamps, 
    const std::vector<Eigen::Matrix<double, 8, 1>> vPriorPose, const WheelCameraCalib &calib, cv::Mat &img, 
    Pose &Twc);

double CalculateScore(const Eigen::Matrix<float, kDescriptorPatchSize, 1> &d1, const Eigen::Matrix<float, kDescriptorPatchSize, 1> &d2);

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

Eigen::Vector3d LogSO3(const Eigen::Matrix3d &R);

Eigen::Matrix3d InverseRightJacobianSO3(const Eigen::Vector3d &v); // BCH近似使用

void VizInteraction(const cv::viz::KeyboardEvent &event, void *b);

double TransformDepthMap2CurrentFrame(KeyFrame *kf1, KeyFrame *kf2, Camera &cam);

bool CheckDepthQuality(const Landmark &lk1, const Pose &T12, const Eigen::Vector2d &p2, const double z);

double CalculatePatchSSD(const KeyFrame *kf1, const KeyFrame *kf2, const Eigen::Vector2i &px1, const Eigen::Vector2i &px2);

void ShowLocalMap(const std::vector<Pose> &vTwc);

enum KeyboardEvent{Reset, StepByStep};

class InteractionParam {
public:
    bool stepBystep = false;
    KeyFrame visualCurF;
    KeyFrame visualCurFinit;
    KeyFrame *visualLastKF = nullptr;
    bool resetWindow = false;
    cv::viz::Viz3d *window; // ("Local Map Viewer"); 
    //cv::Affine3d *viewPose; // 不需要，默认的window会保留现场
    std::set<Landmark*> activePoints;
    std::set<Landmark*> localPoints;
};
extern InteractionParam *interaction;
#endif