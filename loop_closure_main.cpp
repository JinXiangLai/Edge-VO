#include <unistd.h>
#include <fstream>
#include <memory>
#include <random>  // 随机数引擎
#include <stack>
#include <thread>

#include <opencv2/viz/vizcore.hpp>

#include "Config.h"
#include "Initializer.h"
#include "Landmark.h"
#include "Optimizer.h"
#include "Pose.h"
#include "Utils.h"
#include "WheelCameraCalib.h"

#include <opencv2/cudacodec.hpp>  // CUDA 视频编码器

using namespace std;
using namespace cv;

void Run(Optimizer* optimizer);

const cv::Point kViz3DWindowPos(1920 + 1920 / 2,
                                1080 - 100);  // 窗口左上角点在屏幕上的位置

void ResetStatus(Optimizer* optimizer, bool* isInitialized,
                 int* trackLostCount);

int main(int argc, char** argv) {

    // 读取程序参数
    //string configFilePath = "../config.yaml";
    string configFilePath = "../src/tum_config.yaml";

    if (argc < 2) {
        cerr << "[WARNING] Usage: ./main configFile[DEFAULT: " << configFilePath
             << "]" << endl;
    } else {
        configFilePath = string(argv[1]);
        cerr << "[INFO] configFile: " << configFilePath << endl;
    }

    Config _config(configFilePath);
    config = &_config;

    InitColor();

    const int firstImgIdx = config->firstImgIdx;
    // 读取外部数据
    vector<string> vstrImages;
    vector<double> vTimeStamps;
    vector<string> vDepthImgs;
    vector<double> vDepthImgTimes;

    // TODO:需要将轮速系转换为相机系，所以倒不如直接在ORBSLAM3下的框架进行开发呢！！！
    vector<Eigen::Matrix<double, 8, 1>> vPriorPose;
    if (config->model == "pinhole") {
        LoadImages(config->dataDir, vstrImages, vTimeStamps, ".png");
        if (config->useDepthImage || config->initWithTrueDepth ||
            config->debugWithTrueDepthImage) {
            LoadImages(config->dataDir, vDepthImgs, vDepthImgTimes, ".png",
                       true);
        }
    } else {
        LoadImages(config->dataDir, vstrImages, vTimeStamps);
    }
    LoadPriorOdom(config->dataDir, vPriorPose);

    for (size_t i = 1; i < vTimeStamps.size(); ++i) {
        Assert(vTimeStamps[i] - vTimeStamps[i - 1] > 0,
               "Check img timestamp error!!!");
    }
    for (size_t i = 1; i < vPriorPose.size(); ++i) {
        //cout << fixed << vPriorPose[i][0] << " | " << vPriorPose[i-1][0] << endl;
        Assert(vPriorPose[i][0] >= vPriorPose[i - 1][0],
               "Check odom timestamp error!!!");
    }

    auto GetRandColor = []() -> cv::Scalar_<int> {
        return {abs(rand()) % 256, abs(rand()) % 256, abs(rand()) % 256};
    };

    shared_ptr<Camera> cam = make_shared<Camera>(config);
    WheelCameraCalib calib(config->Qcg, config->Pcg, config->wheelRadius);

    vector<KeyFrame> kfVec;

    KeyFrame curKF;
    shared_ptr<LightGlue> lightglue;
    cv::Mat matchImage, matchImgColor;

    const string videoSavePath("./loop_closure_lightglue_match_result.avi");
    cv::Ptr<cv::cudacodec::VideoWriter> writer;
    cv::cuda::GpuMat gpuFrame;

    for (size_t i = firstImgIdx; i < vTimeStamps.size(); ++i) {
        Mat img;
        Pose Twc;
        GetImageAndPose(i, vstrImages, vTimeStamps, vPriorPose, calib, img,
                        Twc);

        cout << i << " th cur img timestamp: " << to_string(vTimeStamps[i])
             << endl;

        KeyFrame curF(img, Twc, cam, i, vTimeStamps[i], config->pyrLevel);

        if (curKF.timestamp_ == 0) {
            curKF = curF;
            curKF.SetTwc(Pose());
            // 初始化匹配器
            lightglue = KeyFrame::lightgluePtr;
            lightglue->SetThreshold(0.05);

            curKF.ExtractSuperpoint();
            const cv::Mat& gray = curKF.grayImg_;

            matchImage = cv::Mat(gray.rows, gray.cols * 2, CV_8UC1);
            matchImgColor = cv::Mat(gray.rows, gray.cols * 2, CV_8UC3);
            gray.copyTo(matchImage(cv::Rect(0, 0, gray.cols, gray.rows)));
            if (!videoSavePath.empty()) {
                writer = cv::cudacodec::createVideoWriter(videoSavePath,
                                                          matchImgColor.size());
            }

            KeyFrame::InitPoseFileMessage();
            continue;
        }

        curF.ExtractSuperpoint();
        const cv::Mat& gray = curF.grayImg_;
        gray.copyTo(matchImage(cv::Rect(gray.cols, 0, gray.cols, gray.rows)));

        Eigen::VectorXf mscores;
        vector<cv::DMatch> lightglueMatches;
        const int matchPairNum =
            lightglue->MatchKeypoints(curKF.kpts_, curF.kpts_, curKF.desc_,
                                      curF.desc_, mscores, lightglueMatches);
        vector<cv::Point2f> pts0(lightglueMatches.size());
        vector<cv::Point2f> pts1(lightglueMatches.size());
        int count = 0;
        cv::cvtColor(matchImage, matchImgColor, cv::COLOR_GRAY2BGR);
        for (const cv::DMatch m : lightglueMatches) {
            const cv::Scalar bgr = GetRandColor();
            const int i = m.queryIdx;
            const int j = m.trainIdx;
            pts0[count] = {curKF.kpts_(i, 0), curKF.kpts_(i, 1)};
            pts1[count] = {curF.kpts_(j, 0), curF.kpts_(j, 1)};
            const cv::Point2f p2 = {pts1[count].x + gray.cols, pts1[count].y};
            cv::circle(matchImgColor, pts0[count], 2, bgr);
            cv::circle(matchImgColor, p2, 2, bgr);
            cv::line(matchImgColor, pts0[count], p2, bgr);
        }

        // 获取匹配点，使用本质矩阵分解及先验尺度计算每个帧的pose
        // curF.setTwc();

        cv::putText(matchImgColor, fmt::format("match num: {}", matchPairNum),
                    cv::Point(10, 30), cv::FONT_ITALIC, 0.80, {0, 0, 255}, 2);

        if (matchPairNum < 150) {
            kfVec.emplace_back(curKF);
            cout << fmt::format("Select {}th img as image0", curF.id_) << endl;
            curKF = std::move(curF);
            gray.copyTo(matchImage(cv::Rect(0, 0, gray.cols, gray.rows)));
        }

        if (!videoSavePath.empty()) {
            gpuFrame.upload(matchImgColor);
            writer->write(gpuFrame);
        }
    }
    return 0;
}

void Run(Optimizer* optimizer) {
    while (!interaction->stopView) {
        if (config->debugShowGlobalMap) {
            interaction->ShowGlobalMapPoint();
        } else if (!optimizer->window_.empty()) {
            optimizer->ShowLocalMap();
        }
        usleep(10 * 1000);
    }
}

void ResetStatus(Optimizer* optimizer, bool* isInitialized,
                 int* trackLostCount) {
    cout << fmt::format("Reset vo system!!!") << endl;
    *isInitialized = false;
    optimizer->window_.clear();
    KeyFrame::Tc0w = Pose();
    KeyFrame::kfOn3Dshow.clear();
    KeyFrame::optFlw.Reset();
    *trackLostCount = 0;
    interaction->trajectory.clear();
}
