#ifndef UTILS
#define UTILS

#include <memory>
#include <unordered_map>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>
#include <Eigen/Dense>

#define CAR_NUM 18
constexpr int kImageWidth = 192;
constexpr int kImageHeight = 108;
constexpr double kRad2Deg = 180/M_PI;
constexpr double kDeg2Rad = M_PI/180;
constexpr int kZnum = 4;
constexpr double kZ[kZnum] = {2.1, 2.2, 2.3, 2.4};

#if CAR_NUM == 15
    // fx cx fy cy
    constexpr double kCameraIntrinsic[4] = {951.728065, 955.374577, 951.7022555, 556.6750875};
    // k1 k2 k3 k4
    constexpr double kCameraDistortion[4] = {-0.024871, -0.018594, 0.015173, -0.005791};
#elif CAR_NUM == 18
    // fx cx fy cy
    constexpr double kCameraIntrinsic[4] = {955.3379635, 961.527827, 955.415214, 541.135945};
    // k1 k2 k3 k4
    constexpr double kCameraDistortion[4] = {-0.026463, -0.018112, 0.016949, -0.007098};
#endif

constexpr double kImageScale = 0.5;
constexpr int kDescriptorPatchLen = 3; // 计算3x3邻域内的描述子
constexpr int kDescriptorPatchSize = kDescriptorPatchLen * kDescriptorPatchLen;
constexpr double kMinTranslation = 0.50; // m, 每隔多少米取1帧图像进行三角化，视差过小时，三角化的距离受噪声影响极大[LSD-SLAM]

// 异常深度滤波操作
constexpr double kMaxDepth = 100.0; // TODO: 距离越大，偏差一个像素可能就会有几十米的偏差
constexpr double kMinDepth = 3.0;
constexpr double kMinGoodTriangulateAngle = 1.5; // 使用深度滤波时不在意视差
constexpr double kMaxGoodTriangulateAngle = 60.0; // 实践过程中发现视差角过大的是异常深度值如1.2m左右

// constexpr double kMaxDescriptorDist = 10.0; // 描述子最大差异值 TODO: 这个值需要仔细设置一下
constexpr int kMaxDescriptorDist = 10; // 描述子最大差异值 TODO: 这个值需要仔细设置一下
constexpr double kThRatio = 1.0; // 为了减小误匹配，必须像ORBSLAM那样设定最优与次优描述子的比值，可能导致特征点稀少
constexpr double kConvergeDiff = 3.0;
constexpr double kAbnormalResidual = 15; // 15 个像素距离为异常残差值

class Pose {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Pose(const Eigen::Quaterniond& q_wb, const Eigen::Vector3d& t_wb);
    Pose() {}
    Pose Inverse() const;
    Eigen::Vector3d operator*(const Eigen::Vector3d &p) const;
    Pose operator*(const Pose& T) const;
    friend std::ostream& operator<<(std::ostream &cout, const Pose& T);
    int Size() const;
    void Update(const Eigen::Vector3d &delta_q, const Eigen::Vector3d &delta_t);

    Eigen::Quaterniond q_wb_ = Eigen::Quaterniond::Identity();
    Eigen::Vector3d t_wb_ = Eigen::Vector3d::Zero();
};

// 定义相机模型
class Camera {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // 实现不考虑畸变参数
    Camera(const double imageScale);
    // TODO：后续我们都将在归一化平面上进行处理
    Eigen::Vector2d Project2PixelPlane(const Eigen::Vector3d &Pc) const;
    Eigen::Vector3d InverseProject(const Eigen::Vector2i &uv, const double &z = 1.0) const;
    std::vector<Eigen::Vector2d> UndistortPoints(std::vector<cv::Point2i> px) const;
    bool InImagePlaneRange(const Eigen::Vector2d &p) const ;

    Eigen::Matrix3d K_ = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d K_inv_ = Eigen::Matrix3d::Identity();
    double fx_, fy_, cx_, cy_;
    double k1_, k2_, k3_, k4_;
    double imageScale_ = kImageScale; // 需要根据图像缩放fx, fy, cx, cy
    const double xStart_, yStart_, xStep_, yStep_, xEnd_, yEnd_; // 归一化平面上的显示范围
};

class Landmark {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Landmark(const Eigen::Vector2d &px, std::shared_ptr<Pose> Twc, const std::shared_ptr<Camera> cam, 
        const double z = 1.0);
    Landmark() {}
    Eigen::Vector3d GetPcNorm() const;
    Eigen::Vector3d GetPc() const;
    Eigen::Vector3d GetPw() const;
    int Size() const; // 优化变量的维度
    void Update(const double delta_z, const bool useInvDepth);
    void UpdateUncertainty();
    bool Converge() const {return uncertainty_ < kConvergeDiff;}

    Eigen::Vector2d uv_; // 像素坐标
    double z_ = 1.0;
    double invZ_ = 1.0;
    // anchor pose
    std::shared_ptr<Pose> Twc_;
    std::shared_ptr<Camera> cam_;
    double depthRange_[2] = {kMinDepth, kMaxDepth};
    double uncertainty_ = kMaxDepth;
    uint64_t descriptor_ = 0;
};

class WheelCameraCalib { 
public:
    WheelCameraCalib();
    Pose Tcv_, Tvc_;
private:
    // 手动填写
#if CAR_NUM == 15
    Eigen::Quaterniond q_cg_{-0.4975894231104555, -0.505161241707068, 0.5085381266442339, -0.4884729876203498};
    Eigen::Vector3d t_cg_{0.1016387795887079, 1.725914890100064, -1.766470201637028};
#elif CAR_NUM == 18
    Eigen::Quaterniond q_cg_{0.4942495072630623, 0.505820516001542, -0.4988584606367882, 0.5010022618843634};
    Eigen::Vector3d t_cg_{0.06400512447560017, 1.695829476938501, -1.788386848651872};
#endif
    double radius_ = 0.376;
};

struct TupleHash {
    size_t operator()(const std::tuple<int, int> &v) const{
        const int v1 = std::get<0>(v);
        const int v2 = std::get<1>(v);
        return (v1 << 1) + (v2 >> 1);
    }
};

class KeyFrame {
public:
    KeyFrame(const cv::Mat &img, const Pose &Twc, std::shared_ptr<Camera> cam, const int level = 1)
    : grayImg_(img)
    , cam_(cam)
    , Twc_ {Twc}
    , level_(level) {
        edgeImg_.resize(level);
        dist_.resize(level);
        dx_.resize(level);
        dy_.resize(level);
        unPx_.resize(level);
    }
    // 可能需要corase2fine的配准
    void CannyEdgeDetect();
    void GenerateDTandDerivative();
    std::vector<Eigen::Vector2d> FindMatches(const Eigen::Vector2d &kp1, const Pose &Twc1);
    size_t GenerateLandmark(KeyFrame &kf1, std::vector<std::vector<Eigen::Vector2d> > &debugGoodKp1, 
        std::vector<std::vector<Eigen::Vector2d> >&debugGoodKp2, const int equalparts);
    void UpdateDepth(const KeyFrame &kf2);
    cv::Mat grayImg_;
    // canny边缘图像已经去畸变了
    std::vector<cv::Mat> edgeImg_, dist_, dx_, dy_;
    std::shared_ptr<Camera> cam_;
    std::vector<Landmark> landmark_;
    Pose Twc_;
    int level_ = 1;
    std::vector<std::vector<Eigen::Vector2d> > unPx_; // 像素平面上的去畸变点
    // std::vector<Eigen::Matrix<float, kDescriptorPatchSize, 1> > descriptor_;
    std::vector<uint64_t> descriptor_; 
    static constexpr int descDim = 63;
    std::unordered_map<std::tuple<int, int>, int, TupleHash> pointMapId_; // 像素坐标与索引的映射
};

// 距离变换是计算前景到背景的距离
cv::Mat GenerateEdgeImage(std::vector<Eigen::Vector2d> &blackPoint);

cv::Mat GetDistanceTransform(cv::Mat img);

std::vector<Eigen::Vector2d> GenerateBlackPoint();

std::vector<Eigen::Vector3d> GeneratePw(const std::vector<Eigen::Vector2d> &blackPoint, const std::shared_ptr<Camera> &cam);

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
    const std::vector<Eigen::Vector2d> &kp2, const std::string &name = "matches", const int ratio = 1, const int jump = 10);

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

void FindImageAndPose(const int idx, const std::vector<std::string> & vstrImages, const std::vector<double> vTimeStamps, 
    const std::vector<Eigen::Matrix<double, 8, 1>> vPriorPose, const WheelCameraCalib &calib, std::vector<cv::Mat> &imgs, 
    std::vector<Pose> &vTwc, const int needNum = 2);

double CalculateScore(const Eigen::Matrix<float, kDescriptorPatchSize, 1> &d1, const Eigen::Matrix<float, kDescriptorPatchSize, 1> &d2);

unsigned long CalculateDescriptor(const cv::Mat &grayImg, const Eigen::Vector2i &px);

int CalculateDescriptorScore(const int v1, const int v2);

void GetProjectRange(const Landmark &lp, const Pose& T21, const Camera &cam, Eigen::Vector2i &xRange, 
    Eigen::Vector2i &yRange);

void ShowPointCloud(const std::vector<Landmark> &ps, const cv::Mat &img);

void ShowPointCloud(const std::vector<Landmark> &ps1, const std::vector<Landmark> &ps2);
#endif