#ifndef CONFIG
#define CONFIG

#include <iostream>
#include <math.h>
#include <string>
#include <vector>


#include <yaml-cpp/yaml.h>

constexpr double kRad2Deg = 180/M_PI;
constexpr double kDeg2Rad = M_PI/180;
constexpr int kDescriptorPatchSize = 9;
enum MessageLevel {Debug, Info, Error};

class Config {
public:
    // 注意： static, extern都只是声明，要在任何函数外进行定义才行
    // 所以，这里只能使用单例模式创建了
    Config(const std::string &yamlFilePath);

    std::string dataDir;
    std::vector<double> intrinstic, distortion;
    std::vector<double> Qcg;
    std::vector<double> Pcg;
    double imageScale;
    double wheelRadius;
    int descriptorPatchLen;
    double minTranslation;
    double minDepth, maxDepth;
    double minGoodTriangulateAngle, maxGoodTriangulateAngle;
    double maxDescriptorDist, maxSSDdist;
    double goodDescriptorDistRatio, goodDescriptorDist;
    double nonMaximumSuppressionRatio;
    double maxTrackProjectPixelError;
    double maxDepthConvergeStd, maxDepthConvergeVariance;
    double abnormalProjectResidual;
    double huberDelta, huberDelta2;
    int keepLastKFnumInWindow, maxKFnumInWindow;
    double imgTimeOffset;
    double needNewKFtrans, needNewKFrot, needNewKFMaxMatchEdgeRatio;
    double filterPixelError;
    int messageLevel;
    int maxIteration;
    int maxActiveLandmarkEachKF;
    int fastTh;
    double relativePoseConstraintWeight;
    std::string model;
    int maxKeepEpilorMatchPointNum = 3;

    bool useInvZ;
    bool showDebugImg;
    int firstImgIdx;
    int loopClosureImgIdx;
    int fastNum;

    double cannyLowerTh, cannyupperTh;
    double minEpipolarSearchLine, maxEpipolarSearchLine;
    double best2SecondRatio;

    int minSearchStep;
    int pyrLevel;
    double SimErrorRatio;
    bool drawGoddEpipolarMatch;
    double best2SecondDist;
    bool drawAllEpipolarMatch;
    int drawEpipolarMatchStartCol;
    double minDepthCompareRatio, maxDepthCompareRatio;
    double maxObvDepthStd, minObvDepthStd;
    int minObvTime;
    double maxDTdistOFkeyPoint;
    int maxFailObvTimeBeforeCreateDepth;
    double minGradientOFkeyPoint;
};

extern Config *config; // 外部可以定义及使用的全局变量，只在main函数初始化一次

constexpr int FASTpoint[16][2] = {
    {0, 3}, {1, 3}, {2, 2}, {3, 1},
    {3, 0}, {3, -1}, {2, -2}, {1, -3},
    {0, -3}, {-1, -3}, {-2, -2}, {-3, -1},
    {-3, 0}, {-3, 1}, {-2, 2}, {-1, 3}
};

#endif
