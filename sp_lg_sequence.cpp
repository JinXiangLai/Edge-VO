//
// Created by haoyuefan on 2021/9/22.
//

#include <chrono>
#include <filesystem>
#include <memory>

#include <opencv2/cudacodec.hpp>  // CUDA 视频编码器

#include "Config.h"
#include "LightGlue.h"
#include "SuperPoint.h"

size_t LoadImages(const std::string& strDirectory,
                  std::vector<std::string>& vstrImages,
                  std::vector<double>& vTimeStamps);

using namespace std;

int main(int argc, char** argv) {

    if (argc < 2) {
        cout << "usage ./sp_lg_sequence yaml_file_path!" << endl;
        return 0;
    }

    Config config(argv[1]);
    string imgDirectory = config.dataDir;

    vector<string> vstrImages;
    vector<double> vTimeStamps;
    LoadImages(imgDirectory, vstrImages, vTimeStamps);
    cout << "Total read image num: " << vstrImages.size() << endl;

    cout << "Building inference engine......" << endl;

    auto superpoint = make_shared<SuperPoint>(config.superpointOnnxFilePath,
                                              config.superpointEngineFilePath);
    if (!superpoint->Build()) {
        cerr << "Error in SuperPoint building engine. Please check your "
                "onnx model path."
             << endl;
        return 0;
    }

    auto lightglue = make_shared<LightGlue>(config.lightglueOnnxFilePath,
                                            config.lightglueEngineFilePath);
    if (!lightglue->Build()) {
        cerr << "Error in lightglue building engine. Please check your "
                "onnx model path."
             << endl;
        return 0;
    }
    lightglue->SetThreshold(0.5);
    cout << "SuperPoint and lightglue inference engine build success." << endl;
    lightglue->ValidateFP16();

    double superpoint_tcount = 0;
    double match_tcount = 0;
    int processNum = 0;

    Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor> kpts0, kpts1;
    Eigen::Matrix<float, Eigen::Dynamic, 256, Eigen::RowMajor> desc0, desc1;
    Eigen::VectorXf mscores;
    vector<cv::DMatch> lightglueMatches;

    cv::Mat image0 = cv::imread(vstrImages[0], cv::IMREAD_GRAYSCALE);
    if (!superpoint->Infer(image0, kpts0, desc0)) {
        cerr << "Failed when extracting features from first image." << endl;
        return 0;
    }

    cv::Mat matchImage = cv::Mat(image0.rows, image0.cols * 2, CV_8UC1);
    cv::Mat matchImgColor = cv::Mat(image0.rows, image0.cols * 2, CV_8UC3);

    image0.copyTo(matchImage(cv::Rect(0, 0, image0.cols, image0.rows)));

    const string videoSavePath("./lightglue_match_result.avi");
    cv::Ptr<cv::cudacodec::VideoWriter> writer;
    cv::cuda::GpuMat gpuFrame;
    if (!videoSavePath.empty()) {
#if CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR < 6
        writer = cv::cudacodec::createVideoWriter(videoSavePath,
                                                  matchImgColor.size(), 24.0);
#else
        writer = cv::cudacodec::createVideoWriter(videoSavePath,
                                                  matchImgColor.size());
#endif
    }

    for (int i = 1; i < vstrImages.size(); ++i) {
        cv::Mat image1 = cv::imread(vstrImages[i], cv::IMREAD_GRAYSCALE);
        if (image0.empty() || image1.empty()) {
            cerr << "Input image is empty. Please check the image path."
                 << endl;
            return 0;
        }

        if (kpts0.rows() == 0) {
        }

        auto start = chrono::high_resolution_clock::now();
        if (!superpoint->Infer(image1, kpts1, desc1)) {
            cerr << "Failed when extracting features from second image."
                 << endl;
            return 0;
        }
        auto end = chrono::high_resolution_clock::now();
        auto duration1 =
            chrono::duration_cast<chrono::milliseconds>(end - start);
        ++processNum;

        start = chrono::high_resolution_clock::now();
        const int matchPairNum = lightglue->MatchKeypoints(
            kpts0, kpts1, desc0, desc1, mscores, lightglueMatches);
        end = chrono::high_resolution_clock::now();
        auto duration2 =
            chrono::duration_cast<chrono::milliseconds>(end - start);
        if (processNum > 1) {
            superpoint_tcount += duration1.count();
            match_tcount += duration2.count();

            cout << fmt::format(
                        "Superpoint infer cost mean time: {:.1f}ms, Lightglue "
                        "infer cost mean time: {:.1f}ms.",
                        superpoint_tcount / (processNum - 1),
                        match_tcount / (processNum - 1))
                 << endl;
        }

        image1.copyTo(
            matchImage(cv::Rect(image0.cols, 0, image0.cols, image0.rows)));

        auto GetRandColor = []() -> cv::Scalar_<int> {
            return {abs(rand()) % 256, abs(rand()) % 256, abs(rand()) % 256};
        };

        // lightglueMatches指示了哪些点应该有连线
        cv::cvtColor(matchImage, matchImgColor, cv::COLOR_GRAY2BGR);
        for (const cv::DMatch m : lightglueMatches) {
            const cv::Scalar bgr = GetRandColor();
            const int i = m.queryIdx;
            const int j = m.trainIdx;
            const cv::Point2f p1(kpts0(i, 0), kpts0(i, 1));
            const cv::Point2f p2(kpts1(j, 0) + image0.cols, kpts1(j, 1));
            cv::circle(matchImgColor, p1, 2, bgr);
            cv::circle(matchImgColor, p2, 2, bgr);
            cv::line(matchImgColor, p1, p2, bgr);
        }

        cv::putText(matchImgColor, fmt::format("match num: {}", matchPairNum),
                    cv::Point(10, 30), cv::FONT_ITALIC, 0.80, {0, 0, 255}, 2);

        if (matchPairNum < 30) {
            cout << fmt::format("Select {}th img as image0", i) << endl;
            kpts0 = kpts1;
            desc0 = desc1;
            image1.copyTo(matchImage(cv::Rect(0, 0, image0.cols, image0.rows)));
        }

        // cv::imwrite("matchImage.png", matchImage);
        //  visualize
        // cv::imshow("matchImgColor", matchImgColor);
        // cv::waitKey(0);
        if (!videoSavePath.empty()) {
            gpuFrame.upload(matchImgColor);
            writer->write(gpuFrame);
        }
    }

    return 0;
}

size_t LoadImages(const string& strDirectory, vector<string>& vstrImages,
                  vector<double>& vTimeStamps) {
    string imageDirectory = strDirectory + "/rgb";
    for (const auto& entry : filesystem::directory_iterator(imageDirectory)) {
        if (entry.is_regular_file()) {  // 仅获取文件，排除子目录
            vstrImages.push_back(entry.path().filename().string());
        }
    }

    sort(vstrImages.begin(), vstrImages.end());
    for (const string& s : vstrImages) {
        vTimeStamps.emplace_back(stod(s.substr(0, s.find_last_of('.'))));
    }
    for (size_t i = 1; i < vTimeStamps.size(); ++i) {
        if (vTimeStamps[i] < vTimeStamps[i - 1]) {
            cout << fmt::format(
                        "Timestamps sort error, "
                        "vTimeStamps{}>vTimeStamps{}<=>{} > {}",
                        i - 1, i, vTimeStamps[i - 1], vTimeStamps[i])
                 << endl;
        }
    }

    for (string& s : vstrImages) {
        s = fmt::format("{}/{}", imageDirectory, s);
    }
    return vstrImages.size();
}
