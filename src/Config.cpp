#include "Config.h"
#include "Utils.h"

using namespace std;

Config::Config(const std::string& yamlFilePath) {

    YAML::Node f = YAML::LoadFile(yamlFilePath);
    if (f.IsNull()) {
        cerr << "Load yaml file at: {" << yamlFilePath << "} Error!" << endl;
        exit(-1);
    }
    cout << "Load yaml file at: {" << yamlFilePath << "}!" << endl;
    dataDir = f["dataDir"].as<string>();
    cout << "dataDir: " << dataDir << endl;
    intrinstic = f["Camera.intrinstic"].as<std::vector<double>>();
    distortion = f["Camera.distortion"].as<std::vector<double>>();
    Qcg = f["Qcg"].as<std::vector<double>>();
    Pcg = f["Pcg"].as<std::vector<double>>();
    imageScale = f["Camera.imageScale"].as<double>();
    wheelRadius = f["wheelRadius"].as<double>();
    descriptorPatchLen = f["descriptorPatchLen"].as<int>();
    minTranslation = f["minTranslation"].as<double>();
    maxDepth = f["maxDepth"].as<double>();
    minDepth = f["minDepth"].as<double>();
    minGoodTriangulateAngle = f["minGoodTriangulateAngle"].as<double>();
    maxGoodTriangulateAngle = f["maxGoodTriangulateAngle"].as<double>();

    maxDescriptorDist =
        f["maxDescriptorDist"].as<double>();  // 可能和int类型比较哦
    goodDescriptorDistRatio = f["goodDescriptorDistRatio"].as<double>();
    goodDescriptorDist = maxDescriptorDist * goodDescriptorDistRatio;

    nonMaximumSuppressionRatio = f["nonMaximumSuppressionRatio"].as<double>();
    maxTrackProjectPixelError = f["maxTrackProjectPixelError"].as<double>();

    maxDepthConvergeStd = f["maxDepthConvergeStd"].as<double>();
    maxDepthConvergeVariance = pow(maxDepthConvergeStd, 2);

    abnormalProjectResidual = f["abnormalProjectResidual"].as<double>();
    huberDelta = f["huberDelta"].as<double>();
    huberDelta2 = huberDelta * huberDelta;
    keepLastKFnumInWindow = f["keepLastKFnumInWindow"].as<int>();
    maxKFnumInWindow = f["maxKFnumInWindow"].as<int>();
    useMarginalization = f["useMarginalization"].as<bool>();
    imgTimeOffset = f["imgTimeOffset"].as<double>();

    needNewKFtrans = f["needNewKFtrans"].as<double>();
    needNewKFrot = f["needNewKFrot"].as<double>();
    maxMeanProjectResidual2CreateKF = f["maxMeanProjectResidual2CreateKF"].as<double>();
    needNewKFMaxMatchEdgeRatio = f["needNewKFMaxMatchEdgeRatio"].as<double>();
    filterPixelError = f["filterPixelError"].as<double>();

    messageLevel = f["messageLevel"].as<int>();
    maxIterationLM = f["maxIterationLM"].as<int>();
    convergeCostDiffLM = f["convergeCostDiffLM"].as<double>();
    maxLambdaValueLM = f["maxLambdaValueLM"].as<double>();
    maxNoImprovementCountLM = f["maxNoImprovementCountLM"].as<int>();
    iterateLogFreqLM = f["iterateLogFreqLM"].as<int>();
    initLambda = f["initLambda"].as<double>();
    maxActiveLandmarkEachKF = f["maxActiveLandmarkEachKF"].as<int>();
    fastTh1 = f["fastTh1"].as<int>();
    fastTh2 = f["fastTh2"].as<int>();
    extractFastNumEachFrame = f["extractFastNumEachFrame"].as<int>();
    maxFlowTrackError = f["maxFlowTrackError"].as<double>();
    optflowWinSize = f["optflowWinSize"].as<int>();
    optflowLayer = f["optflowLayer"].as<int>();
    relativePoseConstraintWeight =
        f["relativePoseConstraintWeight"].as<double>();

    model = f["Camera.model"].as<string>();
    maxKeepEpilorMatchPointNum = f["maxKeepEpilorMatchPointNum"].as<int>();

    useInvZ = f["useInvZ"].as<bool>();
    showDebugImg = f["showDebugImg"].as<bool>();
    firstImgIdx = f["firstImgIdx"].as<int>();
    loopClosureImgIdx = f["loopClosureImgIdx"].as<int>();
    fastNum = f["fastNum"].as<int>();
    cannyLowerTh = f["cannyLowerTh"].as<double>();
    cannyupperTh = f["cannyupperTh"].as<double>();
    maxEpipolarSearchLine = f["maxEpipolarSearchLine"].as<double>();
    minEpipolarSearchLine = f["minEpipolarSearchLine"].as<double>();
    best2SecondRatio = f["best2SecondRatio"].as<double>();
    minSearchStep = f["minSearchStep"].as<int>();
    pyrLevel = f["pyrLevel"].as<int>();
    SimErrorRatio = f["SimErrorRatio"].as<double>();
    drawGoddEpipolarMatch = f["drawGoddEpipolarMatch"].as<bool>();
    best2SecondDist = f["best2SecondDist"].as<double>();
    drawAllEpipolarMatch = f["drawAllEpipolarMatch"].as<bool>();
    drawEpipolarMatchStartCol = f["drawEpipolarMatchStartCol"].as<int>();
    maxSSDdist = pow(maxDescriptorDist, 2) * pow(descriptorPatchLen, 2);

    minDepthCompareRatio = f["minDepthCompareRatio"].as<double>();
    maxDepthCompareRatio = f["maxDepthCompareRatio"].as<double>();

    maxObvDepthStd = f["maxObvDepthStd"].as<double>();
    minObvDepthStd = f["minObvDepthStd"].as<double>();
    minObvTime = f["minObvTime"].as<int>();
    maxDTdistOFkeyPoint = f["maxDTdistOFkeyPoint"].as<double>();
    maxFailObvTimeBeforeCreateDepth =
        f["maxFailObvTimeBeforeCreateDepth"].as<int>();
    minGradientOFkeyPoint = f["minGradientOFkeyPoint"].as<double>();
    useDepthImage = f["useDepthImage"].as<bool>();
    depthFactor = f["depthFactor"].as<double>();
    initWithTrueDepth = f["initWithTrueDepth"].as<bool>();
    debugMessageSaveFolder = f["debugMessageSaveFolder"].as<string>();
    while (debugMessageSaveFolder.back() == '/') {
        debugMessageSaveFolder.pop_back();
    }
    debugWithTrueDepthImage = f["debugWithTrueDepthImage"].as<bool>();
    vector<string> keyPointSet = f["debugKFpointSet"].as<std::vector<string>>();
    for (const string& s : keyPointSet) {
        pixelCount.insert(ParseKeypointSet(s));
    }
    //for (const Eigen::Vector2i& p : pixelCount) {
    //    cout << p.transpose() << endl;
    //}

    debugShowGlobalMap = f["debugShowGlobalMap"].as<bool>();
    debugRunOnDesktop = f["debugRunOnDesktop"].as<bool>();
    debugShowOnlineResult3D = f["debugShowOnlineResult3D"].as<bool>();

    matchNoise = f["matchNoise"].as<double>();
    useAvgDiff = f["useAvgDiff"].as<bool>();
    maxProjectError = f["maxProjectError"].as<double>();
}

Config* config = nullptr;
