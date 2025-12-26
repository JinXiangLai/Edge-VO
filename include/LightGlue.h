
#ifndef LIGHT_GLUE
#define LIGHT_GLUE

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <fmt/core.h>
#include <Eigen/Core>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>

#include "../3rdparty/tensorrtbuffer/include/buffers.h"

#include <cuda_fp16.h>

using tensorrt_common::TensorRTUniquePtr;

#define USE_FP16_PRECISION 0

namespace LightGlueConfig {
constexpr const char* kInputTensorNames[4] = {"kpts0", "kpts1", "desc0",
                                              "desc1"};
constexpr const char* kOutputTensorNames[1] = {"mscores0"};

constexpr int kImageHeight = 480;
constexpr int kImageWidth = 640;
constexpr float kMatchThreshold = 0.01;  // 有双向匹配约束，可以设置小
};                                       // namespace LightGlueConfig

class LightGlue {
   public:
    explicit LightGlue(const std::string& onnxFile,
                       const std::string& engineFile = "");

    bool Build();

    // Eigen默认是列优先存储，这里使用行优先
    bool Infer(
        const Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor>& kpts0,
        const Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor>& kpts1,
        const Eigen::Matrix<float, Eigen::Dynamic, 256, Eigen::RowMajor>& desc0,
        const Eigen::Matrix<float, Eigen::Dynamic, 256, Eigen::RowMajor>& desc1,
        Eigen::VectorXi& indices0, Eigen::VectorXi& indices1,
        Eigen::VectorXf& mscores);

    int MatchKeypoints(
        const Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor>& kpts0,
        const Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor>& kpts1,
        const Eigen::Matrix<float, Eigen::Dynamic, 256, Eigen::RowMajor>& desc0,
        const Eigen::Matrix<float, Eigen::Dynamic, 256, Eigen::RowMajor>& desc1,
        Eigen::VectorXf& mscores, std::vector<cv::DMatch>& matches,
        bool outlier_rejection = false);

    Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor> NormalizeKeypoints(
        const Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor>& kpts,
        int width, int height);

    void SaveEngine();

    bool DeserializeEngine();

    struct MatchPair {
        int index0;
        int index1;
        float score;

        MatchPair(int idx0, int idx1, float sc)
            : index0(idx0), index1(idx1), score(sc) {}
    };
    /**
     * 从得分矩阵中提取匹配对
     * @param scores 得分矩阵 [num_kpts0, num_kpts1]
     * @param matchability0 关键点0的匹配性得分 [num_kpts0]
     * @param matchability1 关键点1的匹配性得分 [num_kpts1]
     * @param matches 输出的匹配对
     * @param num_kpts0 关键点0的数量
     * @param num_kpts1 关键点1的数量
     */
    void ExtractMatches(const float* scores, std::vector<MatchPair>& matches,
                        int num_kpts0, int num_kpts1);

    /**
     * 快速匹配提取（不使用匹配性得分）
     */
    void ExtractMatchesFast(const float* scores,
                            std::vector<MatchPair>& matches, int num_kpts0,
                            int num_kpts1);

    void SetThreshold(float threshold) { threshold_ = threshold; }
    float GetThreshold() const { return threshold_; }

    bool ValidateFP16();
    void CheckPerformanceCharacteristics();

   private:
    std::vector<int> indices0_;
    std::vector<int> indices1_;
    std::vector<double> mscores_;

    // 输入
    nvinfer1::Dims keypoints_0_dims_{};
    nvinfer1::Dims descriptors_0_dims_{};
    nvinfer1::Dims keypoints_1_dims_{};
    nvinfer1::Dims descriptors_1_dims_{};

    // 输出
    nvinfer1::Dims output_scores_dims_{};

    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::shared_ptr<nvinfer1::IExecutionContext> context_;

    bool ConstructNetwork(
        TensorRTUniquePtr<nvinfer1::IBuilder>& builder,
        TensorRTUniquePtr<nvinfer1::INetworkDefinition>& network,
        TensorRTUniquePtr<nvinfer1::IBuilderConfig>& config,
        TensorRTUniquePtr<nvonnxparser::IParser>& parser) const;

    bool ProcessInput(
        const tensorrt_buffer::BufferManager& buffers,
        const Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor>& kpts0,
        const Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor>& kpts1,
        const Eigen::Matrix<float, Eigen::Dynamic, 256, Eigen::RowMajor>& desc0,
        const Eigen::Matrix<float, Eigen::Dynamic, 256, Eigen::RowMajor>&
            desc1);

    /**
     * 计算互最近邻匹配
     */
    void MutualNearestNeighbor(const float* scores, std::vector<int>& matches0,
                               std::vector<int>& matches1,
                               std::vector<float>& scores0,
                               std::vector<float>& scores1, int num_kpts0,
                               int num_kpts1);
    /**
     * 应用得分阈值过滤
     */
    void ApplyThreshold(std::vector<int>& matches0, std::vector<int>& matches1,
                        std::vector<float>& scores0,
                        std::vector<float>& scores1, float threshold);
    bool ProcessOutput(const tensorrt_buffer::BufferManager& buffers,
                       Eigen::VectorXi& indices0, Eigen::VectorXi& indices1,
                       Eigen::VectorXf& mscores);

    double threshold_ = 0.1;

    std::string onnxFilePath_, engineFilePath_;
};

typedef std::shared_ptr<LightGlue> LightGluePtr;

#endif  //SUPER_GLUE_H_
