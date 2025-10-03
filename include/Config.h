#ifndef CONFIG
#define CONFIG

#include <math.h>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>

constexpr double kRad2Deg = 180 / M_PI;
constexpr double kDeg2Rad = M_PI / 180;
constexpr int kDescriptorPatchSize = 9;
enum MessageLevel { Debug, Info, Error };

// 更健壮的哈希函数，避免冲突
struct RobustVector2iHash {
    std::size_t operator()(const Eigen::Vector2i& vec) const {
        // 使用boost的hash_combine思想
        std::size_t seed = 0;
        seed ^=
            std::hash<int>()(vec.x()) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^=
            std::hash<int>()(vec.y()) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

struct Vector2iEqual {
    bool operator()(const Eigen::Vector2i& lhs,
                    const Eigen::Vector2i& rhs) const {
        return lhs == rhs;  // Eigen已经重载了==操作符
    }
};

class Config {
   public:
    // 注意： static, extern都只是声明，要在任何函数外进行定义才行
    // 所以，这里只能使用单例模式创建了
    Config(const std::string& yamlFilePath);

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
    bool useDepthImage;
    double depthFactor;
    bool initWithTrueDepth;
    std::string debugMessageSaveFolder;
    bool debugWithTrueDepthImage;
    std::unordered_set<Eigen::Vector2i, RobustVector2iHash, Vector2iEqual>
        pixelCount;
    bool debugShowGlobalMap;
    bool debugRunOnDesktop;
    bool debugShowOnlineResult3D;
    double matchNoise;
    bool useAvgDiff;
};

extern Config* config;  // 外部可以定义及使用的全局变量，只在main函数初始化一次

constexpr int FASTpoint[16][2] = {
    {0, 3},  {1, 3},   {2, 2},   {3, 1},   {3, 0},  {3, -1}, {2, -2}, {1, -3},
    {0, -3}, {-1, -3}, {-2, -2}, {-3, -1}, {-3, 0}, {-3, 1}, {-2, 2}, {-1, 3}};

#endif
