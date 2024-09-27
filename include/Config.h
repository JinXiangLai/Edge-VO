#ifndef CONFIG
#define CONFIG

#include <iostream>
#include <math.h>

constexpr double kRad2Deg = 180/M_PI;
constexpr double kDeg2Rad = M_PI/180;

#define CAR_NUM 18
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

// TODO: 未进行缩放时，可能存在特征点过多的优化问题过于庞大导致的内存爆炸
constexpr double kImageScale = 0.5;
constexpr int kDescriptorPatchLen = 3; // 计算3x3邻域内的描述子
constexpr int kDescriptorPatchSize = kDescriptorPatchLen * kDescriptorPatchLen;
constexpr double kMinTranslation = 0.5; // m, 每隔多少米取1帧图像进行三角化，视差过小时，三角化的距离受噪声影响极大[LSD-SLAM]

// 异常深度滤波操作
constexpr double kMaxDepth = 100.0; // TODO: 距离越大，偏差一个像素可能就会有几十米的偏差
constexpr double kMinDepth = 3.0;
constexpr double kMinGoodTriangulateAngle = 1.5; // 使用深度滤波时不在意视差
constexpr double kMaxGoodTriangulateAngle = 60.0; // 实践过程中发现视差角过大的是异常深度值如1.2m左右

// constexpr double kMaxDescriptorDist = 10.0; // 描述子最大差异值 TODO: 这个值需要仔细设置一下
constexpr int kMaxDescriptorDist = 10; // 描述子最大差异值 TODO: 这个值需要仔细设置一下
constexpr int kGoodDescriptorDist = kMaxDescriptorDist * 0.6;
constexpr double kMaxTrackProjectError = 1.0; // 追踪landmark不能超过的重投影像素误差

constexpr double kThRatio = 1.0; // 为了减小误匹配，必须像ORBSLAM那样设定最优与次优描述子的比值，可能导致特征点稀少
constexpr double kConvergeDiff = 0.5;
constexpr double kAbnormalResidual = 5; // 15 个像素距离为异常残差值

constexpr int kMaxKFnumInWindow = 7;

constexpr double kImgTimeOffset = 0.06; // 0.06; // sec

constexpr double kNewKFtrans = 2.5; // m
constexpr double kNewKFrot = 10; // degree
constexpr double kNewKFMinMatchEdgeRatio = 0.5; // 重投影重复边缘点
constexpr double kPixelError = 1; // 1, 默认按1个像素误差推导

#endif
