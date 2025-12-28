#ifndef UTILS
#define UTILS

#include <fmt/core.h>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>

#include "Camera.h"
#include "Config.h"
#include "KeyFrame.h"
#include "Landmark.h"
#include "Pose.h"
#include "WheelCameraCalib.h"

#define USE_SSD
#define Undistort  // 进行特征匹配时，需要在未去畸变的图像上进行，但是当三角化时，需要在归一化平面上去畸变
#define USE_INV_DEPTH

constexpr double kMinSceneDepthInCamera = 0.01;      // meter
constexpr double kMinParallaxAng = 30.0 * kDeg2Rad;  // rad
const double kMaxCosValue = cos(kMinParallaxAng);

typedef Eigen::Matrix<double, 3, Eigen::Dynamic> DynamicPointMatrix;

inline const std::map<std::string, cv::Vec3b, std::less<>> kColor = {
    {"red", {0, 0, 255}},       {"green", {0, 255, 0}},
    {"blue", {255, 0, 0}},      {"white", {255, 255, 255}},
    {"yellow", {0, 255, 255}},  {"orange", {0, 165, 255}},
    {"purple", {226, 43, 138}}, {"pink", {255, 0, 255}}};

class KeyFrame;
class Landmark;

enum COLOR { red, orange, yellow, green, blue, purple, pink };

extern std::map<int, cv::Vec3b> Color;

void InitColor();
// 距离变换是计算前景到背景的距离

cv::Mat GetDistanceTransform(cv::Mat img);

std::vector<Eigen::Vector3d> TransformPoint2Pc(
    const Pose& T, std::vector<Eigen::Vector3d>& ps);

// template<typename  C>
// void CaculateDerivative(const cv::Mat &dist, cv::Mat &dx, cv::Mat &dy);
template <typename C>
void CaculateDerivative(const cv::Mat& dist, cv::Mat& dx, cv::Mat& dy) {
    const int h = dist.rows;
    const int w = dist.cols;

    // 差分肯定是float类型
    dx = cv::Mat(h, w, CV_32FC1, 0.);
    dy = dx.clone();

    const C* data = dist.ptr<C>();
    float* datax = dx.ptr<float>();
    float* datay = dy.ptr<float>();

    for (int i = 0; i < h - 1; ++i) {
        // 遍历一行
        for (int j = 1; j < w - 1; ++j) {
            //  dx.at<C>(i, j) = 0.5 * (dist.at<C>(i, j+1) - dist.at<C>(i, j-1));
            datax[i * w + j] =
                0.5 * (data[i * w + j + 1] - data[i * w + j - 1]);
            //dx.at<float>(i, j) = (dist.at<float>(i, j+1) - dist.at<float>(i, j));
        }
    }
    for (int j = 0; j < w - 1; ++j) {
        // 遍历一列
        for (int i = 1; i < h - 1; ++i) {
            //  dy.at<C>(i, j) = 0.5 * (dist.at<C>(i+1, j) - dist.at<C>(i-1, j));
            //dy.at<float>(i, j) = (dist.at<float>(i+1, j) - dist.at<float>(i, j));
            datay[i * w + j] =
                0.5 * (data[(i + 1) * w + j] - data[(i - 1) * w + j]);
        }
    }
}

inline bool InRange(const cv::Mat& img, const Eigen::Vector2i& p) {
    const double imgScale = config->imageScale;
    int jumpPxNum = 6;
#ifdef Undistort
    // jumpPxNum = 26;
#endif
    return p.x() >= jumpPxNum * imgScale &&
           p.x() < img.cols - jumpPxNum * imgScale &&
           p.y() >= jumpPxNum * imgScale &&
           p.y() < img.rows - jumpPxNum * imgScale;  // 把车头像素滤掉
}

Eigen::Matrix3d SkewSymmetric(const Eigen::Vector3d& v);

Eigen::Vector3d SkewSymmetric2Vector(const Eigen::Matrix3d m);

template <typename T>
double BilinearInterpolate(const cv::Mat& img, const Eigen::Vector2d& p) {
    if (!InRange(img, p.cast<int>())) {
        return 0;
    }

    const int col = img.cols;
    const int row = img.rows;
    const int x = int(p.x());
    const int y = int(p.y());
    if (x == col - 1 || x == 0 || y == row - 1 || y == 0 || 1) {
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
    float v2 = img.at<T>(y, x + 1);
    float v3 = img.at<T>(y + 1, x);
    float v4 = img.at<T>(y + 1, x + 1);
    const double wx = p.x() - x;
    const double wy = p.y() - y;
    const double w1 = (1 - wx) * (1 - wy);
    const double w2 = wx * (1 - wy);
    const double w3 = (1 - wx) * wy;
    const double w4 = wx * wy;
    // cout << "w1+w2+w3+w4: " << (w1+w2+w3+w4) << endl; // equal to 1
    return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
}

std::vector<double> CalculateDescriptor(const cv::Mat& grayImg,
                                        const Eigen::Vector2d& px,
                                        const Eigen::Vector2d& epNorm,
                                        const int len = 5);

double CalculateSSD(const std::vector<double>& v1,
                    const std::vector<double>& v2, double avg1, double avg2,
                    const int desLen);

double CalculateSSD(const std::vector<double>& v1,
                    const std::vector<double>& v2, const int desLen);

template <typename Scalar>
Eigen::Quaternion<Scalar> Exp(const Eigen::Matrix<Scalar, 3, 1>& omega) {
    Scalar theta_sq = omega.squaredNorm();
    Scalar theta = sqrt(theta_sq);
    Scalar half_theta = Scalar(0.5) * theta;

    Scalar imag_factor;
    Scalar real_factor;
    const Scalar ellison = Scalar(1e-10);
    if (theta < ellison) {
        Scalar theta_po4 = theta_sq * theta_sq;
        imag_factor = Scalar(0.5) - Scalar(1.0 / 48.0) * theta_sq +
                      Scalar(1.0 / 3840.0) * theta_po4;
        real_factor = Scalar(1) - Scalar(1.0 / 8.0) * theta_sq +
                      Scalar(1.0 / 384.0) * theta_po4;
    } else {
        Scalar sin_half_theta = sin(half_theta);
        imag_factor = sin_half_theta / theta;
        real_factor = cos(half_theta);
    }

    Eigen::Quaternion<Scalar> q(real_factor, imag_factor * omega.x(),
                                imag_factor * omega.y(),
                                imag_factor * omega.z());
    // q.normalize(); // 这里不能强制给它归一化吗？
    if (abs(q.squaredNorm() - Scalar(1)) > ellison) {
        std::cout << "SO3::exp failed! omega: " << omega.transpose()
                  << " real, img: " << real_factor << ", " << imag_factor
                  << std::endl;
        exit(-1);
    }
    return q;
}

void Assert(bool a, const std::string& s);

void ShowImage(const cv::Mat& img, const std::string& name,
               const bool show = true);

Eigen::Vector3d Quat2RPY(const Eigen::Quaterniond& _q);

std::ostream& operator<<(std::ostream& cout, const Pose& T);

cv::Mat DrawMatch(const cv::Mat& img1, const cv::Mat& img2,
                  const std::vector<Eigen::Vector2i>& kp1,
                  const std::vector<Eigen::Vector2i>& kp2,
                  const std::string& name = "matches", const int ratio = 1,
                  const int jump = 10);

char DrawMatch(const cv::Mat& img1, const cv::Mat& img2,
               const std::vector<Eigen::Vector2i>& trajKp1,
               const std::vector<Eigen::Vector2i>& trajKp2,
               const std::vector<Eigen::Vector2i>& goodKp2,
               const std::string& name = "epipolar matches",
               const int ratio = 1, const int jump = 10);

std::vector<Eigen::Vector2d> FindMatchesWithEpipolarConstraintOnImagePlane(
    const Eigen::Vector2d& kp1, const cv::Mat& edgeImg, const Pose& T21,
    const Camera& cam);

Eigen::Vector3d Triangulate(const Eigen::Vector2d& kp2, const Pose& T21,
                            const Camera& cam);

Eigen::Vector3d Triangulate(const Eigen::Vector2d& kp1,
                            const Eigen::Vector2d& kp2, const Pose& T21,
                            const Camera& cam);

double TriangulateDepth(const Eigen::Vector2d& kp1, const Eigen::Vector2d& kp2,
                        const Pose& T21, const Camera& cam);

Pose ConvertRPYandPostion2Pose(const Eigen::Vector3d& rpy,
                               const Eigen::Vector3d& t,
                               const double deg2rad = kDeg2Rad);

void varifyTriangulate();

std::vector<Eigen::Vector2d> CannyEdgeDetect(const cv::Mat& img,
                                             cv::Mat& edgeImg,
                                             const Camera& cam);

size_t LoadImages(const std::string& strDirectory,
                  std::vector<std::string>& vstrImages,
                  std::vector<double>& vTimeStamps,
                  const std::string& imgSuffix = ".jpg",
                  const bool readDepth = 0);

size_t LoadPriorOdom(const std::string& strDirectory,
                     std::vector<Eigen::Matrix<double, 8, 1>>& vPriorPose);

void FindImageAndPose(const int idx, const std::vector<std::string>& vstrImages,
                      const std::vector<double> vTimeStamps,
                      const std::vector<Eigen::Matrix<double, 8, 1>> vPriorPose,
                      const WheelCameraCalib& calib, std::vector<cv::Mat>& imgs,
                      std::vector<Pose>& vTwc, const int needNum = 2);

void GetImageAndPose(const int idx, const std::vector<std::string>& vstrImages,
                     const std::vector<double> vTimeStamps,
                     const std::vector<Eigen::Matrix<double, 8, 1>> vPriorPose,
                     const WheelCameraCalib& calib, cv::Mat& img, Pose& Twc);

bool GetDepthImage(const double rgbTime,
                   const std::vector<std::string>& vstrImages,
                   const std::vector<double> vTimeStamps, cv::Mat& depth);

double CalculateScore(const Eigen::Matrix<float, kDescriptorPatchSize, 1>& d1,
                      const Eigen::Matrix<float, kDescriptorPatchSize, 1>& d2);

uint64_t CalculateDescriptor(const cv::Mat& grayImg, const Eigen::Vector2i& px);

void ShowPointCloud(const std::vector<std::shared_ptr<Landmark>>& ps);

void ShowPointCloud(const std::vector<std::shared_ptr<Landmark>>& ps1,
                    const std::vector<std::shared_ptr<Landmark>>& ps2,
                    const std::string& windowName = "Point cloud",
                    const double zOffset = 0.0);

double GetOnePixelUncertainty(const Eigen::Vector3d& t12,
                              const Eigen::Vector3d& pc1, const double f);

double GetDepthUncertainty(const Eigen::Vector2d& px2,
                           const Eigen::Vector2d& deltaPix2, const double d,
                           const Camera& cam);

bool NeedNewKF(const KeyFrame* kf, const KeyFrame* f);

void ShowPointCloud(const std::unordered_set<std::shared_ptr<Landmark>>& ps);

bool IsFastPoint(const cv::Mat& gray, const int fastTh1, const cv::Point2i& pt,
                 int& response);

Eigen::Vector3d LogSO3(const Eigen::Matrix3d& R);

Eigen::Matrix3d InverseRightJacobianSO3(
    const Eigen::Vector3d& v);  // BCH近似使用

Eigen::MatrixXd CVmat2Eigen(const cv::Mat& m);

cv::Mat Eigen2CVmat(const Eigen::MatrixXd& eigen_mat);

void VizInteraction(const cv::viz::KeyboardEvent& event, void* b);

double CalculatePatchSSD(const KeyFrame* kf1, const KeyFrame* kf2,
                         const Eigen::Vector2i& px1,
                         const Eigen::Vector2i& px2);

void ShowLocalMap(const std::vector<Pose>& vTwc);

void ShowCameraCone(const std::vector<Pose>& vTwc,
                    const std::vector<cv::Mat>& imgs, const Camera& cam);

// 画边缘点的垂直与平行方向
char DrawPerpendicularAndParallelDirectionOFedge(const cv::Mat& edgeImg,
                                                 const cv::Mat& dxImg,
                                                 const cv::Mat& dyImg);

double GetPositiveDepth(const double invZ);

bool GetHostAndCurFrameObservationDepth(const Eigen::Vector2d& kp1,
                                        const Eigen::Vector2d& kp2,
                                        const Eigen::Matrix3d& invK0,
                                        const Pose& T12, double& idepth1,
                                        double& idepth2);

bool GetHostFrameObservationInvDepth(const Eigen::Vector2d& kp1,
                                     const Eigen::Vector2d& kp2,
                                     const Eigen::Matrix3d& invK0,
                                     const Pose& T12, double& idepth1);

double CalculateVarianceByOffsetPx2(const Eigen::Vector2d& kp1,
                                    const Eigen::Vector2d& kp2,
                                    const Eigen::Vector2d& ep2,
                                    const Eigen::Matrix3d& invK0,
                                    const Pose& T12,
                                    const double offsetRatio = 3.0);

double CalculateVariance(const double& estIdepth1, const Eigen::Vector2d& kp1,
                         const Eigen::Vector2d& kp2, const Pose& T21,
                         const Eigen::Matrix3d& invK, const Eigen::Matrix3d& K);

Eigen::Vector2d CalculateObvWrtIdepth1Jacobian(const Eigen::Matrix3d& Rc2_c1,
                                               const double& rho1,
                                               const Eigen::Vector3d& Pn1,
                                               const Eigen::Vector3d& Pc2,
                                               const Eigen::Matrix3d& K);

bool LandmarkTransformHost(const Landmark& lk1, const Pose& T21,
                           const Eigen::Matrix3d& invK, const double& depth2,
                           double& variance2, int& varianceDecreaseNum,
                           const double obvResidual = 0);

Eigen::Vector2d GetEpipolarLineDirection(const Eigen::Vector3d& Pother2this,
                                         const Eigen::Vector2d& p1,
                                         const Camera& cam);

Eigen::Vector2i ParseKeypointSet(const std::string& s);

bool CheckEpipolarLineDirection(const Eigen::Vector2d& ep2,
                                Eigen::Vector2d& ep1);

std::string GetTriangulatePointName(const Eigen::Vector2d& p);

Pose GetPredictPose(const KeyFrame& last1, const KeyFrame& last2);

inline double ChronoMillisecTimeDuration(
    const std::chrono::steady_clock::time_point& t1,
    const std::chrono::steady_clock::time_point& t2) {
    return std::chrono::duration<double>(t2 - t1).count() * 1e3;
}

bool CalculateSim3PosesT12RANSAC(const DynamicPointMatrix& Pc1,
                                 const DynamicPointMatrix& Pc2, Sim3Pose& sT12,
                                 const int minSet = 3,
                                 const double prob = 0.999,
                                 const double inerProb = 0.5);

int CalculateInnerNum(const DynamicPointMatrix& Pc1,
                      const DynamicPointMatrix& Pc2, const Sim3Pose& sT12,
                      const double diffRatio);

bool CalculateSim3PoseT12(const DynamicPointMatrix& Pc1,
                          const DynamicPointMatrix& Pc2, Sim3Pose& sT12);

bool SelectKeyframeInLoopClosure(std::vector<KeyFrame*>& allKeyframe,
                                 int fixedIndex, int loopClosureIndex,
                                 std::vector<KeyFrame*>& selectResult);

int CalculateLoopClosureSim3PoseAndConstraint(
    const Sim3Pose& relativeSim3T12,
    const std::vector<KeyFrame*>& selectKFresult,
    std::vector<Sim3Pose>& loopClosurePoseTwc,
    std::vector<Sim3Pose>& relativePoseConstraint,
    const bool use_priorTwc = true);

inline double CalculateParallax(const Eigen::Vector2d& p1,
                                const Eigen::Vector2d& p2,
                                const Eigen::Vector2d& c) {
    const Eigen::Vector2d dp = p2 - p1;
    const double parallax = dp.norm();
    const Eigen::Vector2d dir1 = dp.normalized();
    const Eigen::Vector2d dir2 = (p2 - c).normalized();
    const double cosValue = dir1.dot(dir2);
    if (abs(cosValue) < kMaxCosValue) {
        return parallax;
    }

    return -1.0;
}

template <int rows, int cols>
void EmplaceBackTriplet(const int startRow, const int startCol,
                        const Eigen::Matrix<double, rows, cols>& blockH,
                        std::vector<Eigen::Triplet<double>>& triplets) {
    for (int i = 0; i < rows; ++i) {
        const int trueRow = startRow + i;
        for (int j = 0; j < cols; ++j) {
            const int trueCol = startCol + j;
            triplets.emplace_back(trueRow, trueCol, blockH(i, j));
        }
    }
}

template <int rows, int cols>
void UpdateSparseHessianMatrix(const int startRow, const int startCol,
                               const Eigen::Matrix<double, rows, cols>& blockH,
                               std::vector<std::unordered_map<int, double*>>&
                                   colMajorSparseMatrixRowId2DataPtr) {
    for (int i = 0; i < rows; ++i) {
        const int trueRow = startRow + i;
        for (int j = 0; j < cols; ++j) {
            const int trueCol = startCol + j;
            // H.coeffRef(trueRow, trueCol) += blockH(i, j);  // 该方式访问速度过慢
            *(colMajorSparseMatrixRowId2DataPtr[trueCol][trueRow]) +=
                blockH(i, j);
        }
    }
}

template <int rows, int cols>
void UpdateSparseHessianMatrix(
    const int startRow, const int startCol,
    const Eigen::Matrix<double, rows, cols>& blockH,
    std::vector<std::vector<double*>>& colMajorSparseMatrixRowId2DataPtr) {
    for (int i = 0; i < rows; ++i) {
        const int trueRow = startRow + i;
        for (int j = 0; j < cols; ++j) {
            const int trueCol = startCol + j;
            // H.coeffRef(trueRow, trueCol) += blockH(i, j);  // 该方式访问速度过慢
            *(colMajorSparseMatrixRowId2DataPtr[trueCol][trueRow]) +=
                blockH(i, j);
        }
    }
}

template <typename T>
inline double ComputeObv2EpipolarLineDist(const Eigen::Matrix<T, 3, 1>& l2,
                                          const Eigen::Matrix<T, 2, 1>& obv) {
    // l2: ax+by+z=0
    // if (abs(l2.x()) < 1e-10 && abs(l2.y()) < 1e-10) {
    //     return 1e6;
    // } // 外部判定
    return abs(l2.x() * obv.x() + l2.y() * obv.y() + l2.z()) /
           sqrt(l2.x() * l2.x() + l2.y() * l2.y());
}

inline Eigen::Vector3d ComputeEpipolarLine(
    const Pose& transform_2_1, const Eigen::Matrix3d& inv_cam_intrinsic_,
    const Eigen::Vector3d& p1) {
    return inv_cam_intrinsic_.transpose() * SkewSymmetric(transform_2_1.t_wb_) *
           transform_2_1.q_wb_.toRotationMatrix() * inv_cam_intrinsic_ * p1;
}

class InteractionParam {
   public:
    bool stepBystep = false;
    KeyFrame visualCurF;      // 进行pose优化后的当前帧
    KeyFrame visualCurFinit;  // 未进行pose优化前的当前帧
    KeyFrame visualLastKF;
    bool resetWindow = false;
    cv::viz::Viz3d* window;  // ("Local Map Viewer");
    //cv::Affine3d *viewPose; // 不需要，默认的window会保留现场
    std::mutex mutPoints;
    std::unordered_set<std::shared_ptr<Landmark>> activePoints;
    std::unordered_set<std::shared_ptr<Landmark>> localPoints;
    std::vector<Eigen::Vector3d> allMapPoints;
    void ShowGlobalMapPoint();
    std::atomic<bool> stopView = false;
    std::vector<Eigen::Vector3d> trajectory;
    std::string imgSaveFolderPath;
    bool SetImgSaveFolderPath(const std::string& path);
};
extern InteractionParam* interaction;
#endif