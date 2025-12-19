
#include "SuperPoint.h"
#include <memory>
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <utility>

using namespace tensorrt_common;
using namespace tensorrt_log;
using namespace tensorrt_buffer;

using namespace std;
SuperPoint::SuperPoint(const string& onnxFile, const string& engineFile)
    : onnxFilePath_(onnxFile), engineFilePath_(engineFile) {
    if (onnxFilePath_.empty()) {
        cout << "[Warnning] Superpoint onnxFilePath_ is empty!" << endl;
        if (engineFilePath_.empty()) {
            cout << "[Error] Superpoint engineFilePath_ is empty!" << endl;
            exit(-1);
        }
    }

    if (engineFilePath_.empty()) {
        engineFilePath_ = fmt::format(
            "{}.engine",
            onnxFilePath_.substr(0, onnxFilePath_.find_last_of('.')));
    }

    cout << fmt::format("Superpoint onnxFilePath_: {}, engineFilePath_: {}",
                        onnxFilePath_, engineFilePath_)
         << endl;
    setReportableSeverity(Logger::Severity::kINTERNAL_ERROR);
}

bool SuperPoint::Build() {
    if (DeserializeEngine()) {
        return true;
    }
    auto builder = TensorRTUniquePtr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder) {
        return false;
    }
    const auto explicit_batch =
        1U << static_cast<uint32_t>(
            NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = TensorRTUniquePtr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(explicit_batch));
    if (!network) {
        return false;
    }
    auto config = TensorRTUniquePtr<nvinfer1::IBuilderConfig>(
        builder->createBuilderConfig());
    if (!config) {
        return false;
    }
    auto parser = TensorRTUniquePtr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, gLogger.getTRTLogger()));
    if (!parser) {
        return false;
    }

    auto profile = builder->createOptimizationProfile();
    if (!profile) {
        return false;
    }
    profile->setDimensions(SuperPointConfig::kInputTensorName,
                           OptProfileSelector::kMIN, Dims4(1, 1, 480, 640));
    profile->setDimensions(SuperPointConfig::kInputTensorName,
                           OptProfileSelector::kOPT, Dims4(1, 1, 480, 640));
    profile->setDimensions(SuperPointConfig::kInputTensorName,
                           OptProfileSelector::kMAX, Dims4(1, 1, 480, 640));
    config->addOptimizationProfile(profile);

    auto constructed = ConstructNetwork(builder, network, config, parser);
    if (!constructed) {
        return false;
    }
    auto profile_stream = makeCudaStream();
    if (!profile_stream) {
        return false;
    }
    config->setProfileStream(*profile_stream);
    TensorRTUniquePtr<IHostMemory> plan{
        builder->buildSerializedNetwork(*network, *config)};
    if (!plan) {
        return false;
    }
    TensorRTUniquePtr<IRuntime> runtime{
        createInferRuntime(gLogger.getTRTLogger())};
    if (!runtime) {
        return false;
    }
    engine_ = shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()));
    if (!engine_) {
        return false;
    }
    SaveEngine();
    // 检测输入个数
    ASSERT(network->getNbInputs() == 1);

    input_dims_ = network->getInput(0)->getDimensions();
    ASSERT(input_dims_.nbDims == 4);

    // 检测输出个数
    ASSERT(network->getNbOutputs() == 2);

    // 检测得分热力图的输出维度
    semi_dims_ = network->getOutput(0)->getDimensions();
    ASSERT(semi_dims_.nbDims == 3);

    desc_dims_ = network->getOutput(1)->getDimensions();
    ASSERT(desc_dims_.nbDims == 4);
    return true;
}

bool SuperPoint::ConstructNetwork(
    TensorRTUniquePtr<nvinfer1::IBuilder>& builder,
    TensorRTUniquePtr<nvinfer1::INetworkDefinition>& network,
    TensorRTUniquePtr<nvinfer1::IBuilderConfig>& config,
    TensorRTUniquePtr<nvonnxparser::IParser>& parser) const {
    auto parsed = parser->parseFromFile(
        onnxFilePath_.c_str(),
        static_cast<int>(gLogger.getReportableSeverity()));
    if (!parsed) {
        return false;
    }
    config->setMaxWorkspaceSize(512_MiB);
    config->setFlag(BuilderFlag::kFP16);
    return true;
}

bool SuperPoint::Infer(
    const cv::Mat& image,
    Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor>& kpts,
    Eigen::Matrix<float, Eigen::Dynamic, 256, Eigen::RowMajor>& desc) {
    if (!context_) {
        context_ = TensorRTUniquePtr<nvinfer1::IExecutionContext>(
            engine_->createExecutionContext());
        if (!context_) {
            return false;
        }
    }

    assert(engine_->getNbBindings() == 3);

    const int input_index =
        engine_->getBindingIndex(SuperPointConfig::kInputTensorName);

    context_->setBindingDimensions(input_index,
                                   Dims4(1, 1, SuperPointConfig::kImageHeight,
                                         SuperPointConfig::kImageWidth));

    BufferManager buffers(engine_, 0, context_.get());

    if (!ProcessInput(buffers, image)) {
        return false;
    }
    buffers.copyInputToDevice();

    bool status = context_->executeV2(buffers.getDeviceBindings().data());
    if (!status) {
        return false;
    }

    buffers.copyOutputToHost();
    if (!ProcessOutput(buffers, kpts, desc)) {
        return false;
    }
    return true;
}

bool SuperPoint::ProcessInput(const BufferManager& buffers,
                              const cv::Mat& image) {
    // 输入图像的维度
    input_dims_.d[2] = image.rows;
    input_dims_.d[3] = image.cols;

    // 输出得分热力图的维度，已经转到与图像分辨率一致
    semi_dims_.d[1] = image.rows;
    semi_dims_.d[2] = image.cols;

    // 输出描述子的维度
    desc_dims_.d[1] = 256;
    desc_dims_.d[2] = image.rows / 8;
    desc_dims_.d[3] = image.cols / 8;

    // 获取网络的输入向量，即将图像展成1xn维
    auto* host_data_buffer = static_cast<float*>(
        buffers.getHostBuffer(SuperPointConfig::kInputTensorName));

    // 将像素值归一化到[0, 1]
    for (int row = 0; row < image.rows; ++row) {
        for (int col = 0; col < image.cols; ++col) {
            host_data_buffer[row * image.cols + col] =
                float(image.at<unsigned char>(row, col)) / 255.0;
        }
    }
    return true;
}

void SuperPoint::FindHighScoreIndex(vector<float>& scores,
                                    vector<vector<int>>& keypoints, int h,
                                    int w, double threshold) {
    const float heightBorder = h - SuperPointConfig::kRemoveBorder;
    const float widthBorder = w - SuperPointConfig::kRemoveBorder;
    auto PointInBorder = [&heightBorder, &widthBorder](const int x,
                                                       const int y) -> bool {
        return x > SuperPointConfig::kRemoveBorder && x < widthBorder &&
               y > SuperPointConfig::kRemoveBorder && y < heightBorder;
    };

    vector<float> sortScores = scores;
    sort(sortScores.begin(), sortScores.end(),
         [](const float a, const float b) { return a > b; });
    const float kMinUsefulThreshold = 1e-6;
    if (sortScores.size() > SuperPointConfig::kMaxKeypoints) {
        if (sortScores[SuperPointConfig::kMaxKeypoints] > kMinUsefulThreshold) {
            threshold = sortScores[SuperPointConfig::kMaxKeypoints];
        } else {
            vector<float>::iterator it = sortScores.end();
            while (it != sortScores.begin()) {
                --it;
                if (*it > kMinUsefulThreshold) {
                    threshold = *it;
                    break;
                }
            }
        }
    } else {
        threshold = kMinUsefulThreshold;
    }

#if 1
    vector<float> new_scores;
    for (int i = 0; i < scores.size(); ++i) {
        if (scores[i] > threshold) {
            // 将1维得分索引转为图像，转换关系为 i = w*row + col
            vector<int> location = {int(i % w), i / w};
            if (PointInBorder(location[0], location[1])) {
                keypoints.emplace_back(location);
                new_scores.push_back(scores[i]);
            }
        }
    }
#else
    // 将图像划分成网格，每个网格内提取响应值最大的点，
    // superpoint似乎不支持这样做，效果不好
    // 保证至少能提取到kMaxKeypoints个特征点，使得推理引擎能够一直使用1024个点
    const float kMinScore = -1.0;
    vector<float> new_scores;
    new_scores.reserve(2000);
    keypoints.reserve(2000);
    for (int topX = 0; topX < w; topX += eachGridSize_.width) {
        for (int topY = 0; topY < h; topY += eachGridSize_.height) {
            float bestScore = kMinScore;
            int bestIndex = -1;
            const int downX = min(topX + eachGridSize_.width, w);
            const int downY = min(topY + eachGridSize_.height, h);
            for (int j = topX; j < downX; ++j) {
                for (int i = topY; i < downY; ++i) {
                    const int index = i * w + j;
                    if (scores[index] >= threshold) {
                        // 将响应值大的点直接添加进来，避免lightglue匹配失败
                        vector<int> location = {j, i};
                        if (PointInBorder(location[0], location[1])) {
                            new_scores.emplace_back(scores[index]);
                            keypoints.emplace_back(
                                location);  // 这里直接填充(x, y)
                        }
                        // continue;
                    }
                    // 控制阈值，避免重复添加
                    if (scores[index] < threshold &&
                        scores[index] > bestScore) {
                        bestScore = scores[index];
                        bestIndex = index;
                    }
                }
            }

            if (bestIndex >= 0) {
                // 获取到网格内的最佳响应点
                vector<int> location = {int(bestIndex % w), bestIndex / w};
                if (PointInBorder(location[0], location[1])) {
                    new_scores.emplace_back(bestScore);
                    keypoints.emplace_back(location);
                }
            }
        }
    }
#endif

    scores.swap(new_scores);
    cout << fmt::format(
        "Superpoint extract {} kpts, by eachGridSize: ({}, {}).\n",
        scores.size(), eachGridSize_.width, eachGridSize_.height);
    // for (int i = 0; i < scores.size(); ++i) {
    //     cout << fmt::format("scores[{}]: {}, kp: ({}, {}), w: {}, h: {}\n", i,
    //                         scores[i], keypoints[i][0], keypoints[i][1], w, h);
    // }
}

void SuperPoint::RemoveBorders(vector<vector<int>>& keypoints,
                               vector<float>& scores, int border, int height,
                               int width) {
    vector<vector<int>> keypoints_selected;
    vector<float> scores_selected;
    keypoints_selected.reserve(keypoints.size());
    scores_selected.reserve(scores.size());
    const float heightBorder = height - border;
    const float widthBorder = width - border;
    for (int i = 0; i < keypoints.size(); ++i) {
        bool flag_h =
            (keypoints[i][1] >= border) && (keypoints[i][1] < heightBorder);
        bool flag_w =
            (keypoints[i][0] >= border) && (keypoints[i][0] < widthBorder);
        if (flag_h && flag_w) {
            keypoints_selected.emplace_back(keypoints[i]);
            scores_selected.emplace_back(scores[i]);
        }
    }
    keypoints.swap(keypoints_selected);
    scores.swap(scores_selected);
}

vector<size_t> SuperPoint::SortIndexes(vector<float>& data) {
    vector<size_t> indexes(data.size());
    iota(indexes.begin(), indexes.end(), 0);
    sort(indexes.begin(), indexes.end(),
         [&data](size_t i1, size_t i2) { return data[i1] > data[i2]; });
    return indexes;
}

void SuperPoint::TopKkeypoints(vector<vector<int>>& keypoints,
                               vector<float>& scores, int k) {
    if (k < keypoints.size() && k != -1) {
        vector<vector<int>> keypoints_top_k(SuperPointConfig::kMaxKeypoints);
        vector<float> scores_top_k(SuperPointConfig::kMaxKeypoints);
        // 返回得分从大到小的索引值
        vector<size_t> indexes = SortIndexes(scores);
        for (int i = 0; i < SuperPointConfig::kMaxKeypoints; ++i) {
            keypoints_top_k[i] = keypoints[indexes[i]];
            scores_top_k[i] = scores[indexes[i]];
        }
        keypoints.swap(keypoints_top_k);
        scores.swap(scores_top_k);
    }
}

void NormalizeKeypoints(const vector<vector<int>>& keypoints,
                        vector<vector<float>>& keypoints_norm, int h, int w,
                        int s) {
    // 得分图是原图分辨率，而描述子图是1/8分辨率
    for (auto& keypoint : keypoints) {
        // 将图像坐标转到以描述子热力图为中心的坐标，这里s取值应该是8
        vector<float> kp = {static_cast<float>(keypoint[0] - s / 2 + 0.5),
                            static_cast<float>(keypoint[1] - s / 2 + 0.5)};
        // 归一化坐标值至(0, 1)
        kp[0] = kp[0] / (w * s - s / 2 - 0.5);
        kp[1] = kp[1] / (h * s - s / 2 - 0.5);

        // 归一化到-1, 1]以便查询对应关键点的描述子热力图
        kp[0] = kp[0] * 2 - 1;
        kp[1] = kp[1] * 2 - 1;
        keypoints_norm.push_back(kp);
    }
}

int Clip(int val, int max) {
    if (val < 0)
        return 0;
    return min(val, max - 1);
}

void GridSample(const float* input, vector<vector<float>>& grid,
                vector<vector<float>>& output, int dim, int h, int w) {
    // descriptors 1, 256, image_height/8, image_width/8
    // keypoints 1, 1, number, 2
    // out 1, 256, 1, number
    for (auto& g : grid) {
        float ix = ((g[0] + 1) / 2) * (w - 1);
        float iy = ((g[1] + 1) / 2) * (h - 1);

        int ix_nw = Clip(floor(ix), w);
        int iy_nw = Clip(floor(iy), h);

        int ix_ne = Clip(ix_nw + 1, w);
        int iy_ne = Clip(iy_nw, h);

        int ix_sw = Clip(ix_nw, w);
        int iy_sw = Clip(iy_nw + 1, h);

        int ix_se = Clip(ix_nw + 1, w);
        int iy_se = Clip(iy_nw + 1, h);

        float nw = (ix_se - ix) * (iy_se - iy);
        float ne = (ix - ix_sw) * (iy_sw - iy);
        float sw = (ix_ne - ix) * (iy - iy_ne);
        float se = (ix - ix_nw) * (iy - iy_nw);

        // 取出一个256维的描述子
        vector<float> descriptor;
        for (int i = 0; i < dim; ++i) {
            // 256x60x106 dhw
            // x * height * depth + y * depth + z
            float nw_val = input[i * h * w + iy_nw * w + ix_nw];
            float ne_val = input[i * h * w + iy_ne * w + ix_ne];
            float sw_val = input[i * h * w + iy_sw * w + ix_sw];
            float se_val = input[i * h * w + iy_se * w + ix_se];
            descriptor.push_back(nw_val * nw + ne_val * ne + sw_val * sw +
                                 se_val * se);
        }
        output.push_back(descriptor);
    }
}

template <typename Iter_T>
double VectorNormalize(Iter_T first, Iter_T last) {
    return sqrt(inner_product(first, last, first, 0.0));
}

void NormalizeDescriptors(vector<vector<float>>& dest_descriptors) {
    // 将描述子进行归一化
    for (auto& descriptor : dest_descriptors) {
        // 求描述子向量范数的倒数
        float norm_inv =
            1.0 / VectorNormalize(descriptor.begin(), descriptor.end());
        // 描述子每位都除以范数
        transform(descriptor.begin(), descriptor.end(), descriptor.begin(),
                  bind1st(multiplies<float>(), norm_inv));
    }
}

void SuperPoint::SampleDescriptors(vector<vector<int>>& keypoints,
                                   float* descriptors,
                                   vector<vector<float>>& dest_descriptors,
                                   int dim, int h, int w, int s) {
    // dim应该是superpoint描述子维度256
    vector<vector<float>> keypoints_norm;
    // 将像素坐标下的关键点转到[-1, 1]范围，以便查询对应的描述子热力图，
    // 注意，关键点 keypoints{{x0,y0}, {x1,y1}...}仍是图像坐标系下
    NormalizeKeypoints(keypoints, keypoints_norm, h, w, s);
    // 根据归一化的关键点坐标查询到对应的描述子
    GridSample(descriptors, keypoints_norm, dest_descriptors, dim, h, w);
    // 将描述子L2范数归一化
    NormalizeDescriptors(dest_descriptors);
}

bool SuperPoint::ProcessOutput(
    const BufferManager& buffers,
    Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor>& kpts,
    Eigen::Matrix<float, Eigen::Dynamic, 256, Eigen::RowMajor>& desc) {
    keypoints_.clear();
    descriptors_.clear();
    // tensor: float32[1,num_keypoints]
    auto* output_score = static_cast<float*>(
        buffers.getHostBuffer(SuperPointConfig::kOutputTensorNames[0]));
    // tensor: float32[Transposedescriptors_dim_0,num_keypoints,Transposedescriptors_dim_2]
    auto* output_desc = static_cast<float*>(
        buffers.getHostBuffer(SuperPointConfig::kOutputTensorNames[1]));

    // superpoint中，输出的得分是一张热力图，同时对应的描述子也是一张热力图
    int semi_feature_map_h = semi_dims_.d[1];
    int semi_feature_map_w = semi_dims_.d[2];
    // 得分数组转为vector形式
    vector<float> scores_vec(
        output_score, output_score + semi_feature_map_h * semi_feature_map_w);
    // 这里是对得分热力图超过阈值的点位置进行保存，
    // 在这里，keypoints_存储了像素坐标
    FindHighScoreIndex(scores_vec, keypoints_, semi_feature_map_h,
                       semi_feature_map_w,
                       SuperPointConfig::kKeypointThreshold);
    // 移除处于得分热力图边缘的特征点
    // RemoveBorders(keypoints_, scores_vec, SuperPointConfig::kRemoveBorder,
    //               semi_feature_map_h, semi_feature_map_w);
    // 保留得分最大的k关键点，保证每次输入lightglue的点数量都是一致的，以降低耗时
    TopKkeypoints(keypoints_, scores_vec, SuperPointConfig::kMaxKeypoints);

    kpts.resize(scores_vec.size(), 2);
    desc.resize(scores_vec.size(), 256);
    // 对应于256 x H/8 x W/8
    int desc_feature_dim = desc_dims_.d[1];
    int desc_feature_map_h = desc_dims_.d[2];
    int desc_feature_map_w = desc_dims_.d[3];
    SampleDescriptors(keypoints_, output_desc, descriptors_, desc_feature_dim,
                      desc_feature_map_h, desc_feature_map_w);

    // 将关键点图像坐标和其对应的描述子存储下来
    for (int i = 0; i < keypoints_.size(); ++i) {
        kpts(i, 0) = keypoints_[i][0];
        kpts(i, 1) = keypoints_[i][1];
    }

    for (int i = 0; i < descriptors_.size(); ++i) {
        for (int j = 0; j < 256; ++j) {
            desc(i, j) = descriptors_[i][j];
        }
    }
    return true;
}

void SuperPoint::Visualization(const string& image_name, const cv::Mat& image) {
    cv::Mat image_display;
    if (image.channels() == 1)
        cv::cvtColor(image, image_display, cv::COLOR_GRAY2BGR);
    else
        image_display = image.clone();
    for (auto& keypoint : keypoints_) {
        cv::circle(image_display, cv::Point(int(keypoint[0]), int(keypoint[1])),
                   1, cv::Scalar(255, 0, 0), -1, 16);
    }
    cv::imwrite(image_name + ".jpg", image_display);
}

void SuperPoint::SaveEngine() {
    if (engineFilePath_.empty())
        return;
    if (engine_ != nullptr) {
        nvinfer1::IHostMemory* data = engine_->serialize();
        ofstream file(engineFilePath_, ios::binary);
        if (!file)
            return;
        file.write(reinterpret_cast<const char*>(data->data()), data->size());
    }
}

bool SuperPoint::DeserializeEngine() {
    ifstream file(engineFilePath_, ios::binary);
    if (file.is_open()) {
        file.seekg(0, ifstream::end);
        size_t size = file.tellg();
        file.seekg(0, ifstream::beg);
        char* model_stream = new char[size];
        file.read(model_stream, size);
        file.close();
        IRuntime* runtime = createInferRuntime(gLogger);
        if (runtime == nullptr) {
            delete[] model_stream;
            return false;
        }
        engine_ = shared_ptr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(model_stream, size));
        if (engine_ == nullptr) {
            delete[] model_stream;
            return false;
        }
        delete[] model_stream;
        return true;
    }
    return false;
}
