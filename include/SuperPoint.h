//
// Created by haoyuefan on 2021/9/22.
//

#ifndef SUPER_POINT
#define SUPER_POINT

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <fmt/core.h>
#include <Eigen/Core>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>


#include "../3rdparty/tensorrtbuffer/include/buffers.h"

using tensorrt_common::TensorRTUniquePtr;

namespace SuperPointConfig {
constexpr int kImageHeight = 480;
constexpr int kImageWidth = 640;
constexpr const char* kInputTensorName = "image";
constexpr const char* kOutputTensorNames[2] = {"scores", "descriptors"};

// 可调试配置
constexpr int kMaxKeypoints = 1024;
constexpr double kKeypointThreshold = 0.005;
constexpr int kRemoveBorder = 4;  // 排除边缘位置的高得分点
};                                // namespace SuperPointConfig

class SuperPoint {
   public:
    explicit SuperPoint(const std::string& onnxFile,
                        const std::string& engineFile = "");

    bool Build();

    bool Infer(
        const cv::Mat& image,
        Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor>& kpts,
        Eigen::Matrix<float, Eigen::Dynamic, 256, Eigen::RowMajor>& desc);

    void Visualization(const std::string& image_name, const cv::Mat& image);

    void SaveEngine();

    bool DeserializeEngine();

   private:
    nvinfer1::Dims input_dims_{};
    nvinfer1::Dims semi_dims_{};
    nvinfer1::Dims desc_dims_{};
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::shared_ptr<nvinfer1::IExecutionContext> context_;
    std::vector<std::vector<int>>
        keypoints_;  // 存储了像素坐标列表[(x, y), ()...]，TODO：改成Eigen::Vector2i存储
    std::vector<std::vector<float>> descriptors_;

    bool ConstructNetwork(
        TensorRTUniquePtr<nvinfer1::IBuilder>& builder,
        TensorRTUniquePtr<nvinfer1::INetworkDefinition>& network,
        TensorRTUniquePtr<nvinfer1::IBuilderConfig>& config,
        TensorRTUniquePtr<nvonnxparser::IParser>& parser) const;

    bool ProcessInput(const tensorrt_buffer::BufferManager& buffers,
                      const cv::Mat& image);

    bool ProcessOutput(
        const tensorrt_buffer::BufferManager& buffers,
        Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor>& kpts,
        Eigen::Matrix<float, Eigen::Dynamic, 256, Eigen::RowMajor>& desc);

    void RemoveBorders(std::vector<std::vector<int>>& keypoints,
                       std::vector<float>& scores, int border, int height,
                       int width);

    std::vector<size_t> SortIndexes(std::vector<float>& data);

    void TopKkeypoints(std::vector<std::vector<int>>& keypoints,
                       std::vector<float>& scores, int k);

    void FindHighScoreIndex(std::vector<float>& scores,
                            std::vector<std::vector<int>>& keypoints, int h,
                            int w, double threshold);

    void SampleDescriptors(std::vector<std::vector<int>>& keypoints,
                           float* descriptors,
                           std::vector<std::vector<float>>& dest_descriptors,
                           int dim, int h, int w, int s = 8);

    std::string onnxFilePath_, engineFilePath_;
};

typedef std::shared_ptr<SuperPoint> SuperPointPtr;

#endif  //SUPER_POINT
