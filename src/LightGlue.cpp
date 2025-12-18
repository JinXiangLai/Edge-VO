
#include "LightGlue.h"
#include <cfloat>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <utility>
#include "SuperPoint.h"

using namespace tensorrt_common;
using namespace tensorrt_log;
using namespace tensorrt_buffer;
using namespace std;

LightGlue::LightGlue(const std::string& onnxFile, const std::string& engineFile)
    : onnxFilePath_(onnxFile), engineFilePath_(engineFile) {
    if (onnxFilePath_.empty()) {
        cout << "[Warnning] LightGlue onnxFilePath_ is empty!" << endl;
        if (engineFilePath_.empty()) {
            cout << "[Error] LightGlue engineFilePath_ is empty!" << endl;
            exit(-1);
        }
    }

    if (engineFilePath_.empty()) {
        engineFilePath_ = fmt::format(
            "{}.engine",
            onnxFilePath_.substr(0, onnxFilePath_.find_last_of('.')));
    }

    cout << fmt::format("LightGlue onnxFilePath_: {}, engineFilePath_: {}",
                        onnxFilePath_, engineFilePath_)
         << endl;
    setReportableSeverity(Logger::Severity::kINTERNAL_ERROR);
}

bool LightGlue::Build() {
    // 如果已有engine，则返回成功，不再重复构建
    if (DeserializeEngine()) {
        return true;
    }

    auto builder = TensorRTUniquePtr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder) {
        cout << "error 1" << endl;
        return false;
    }

    const auto explicit_batch =
        1U << static_cast<uint32_t>(
            NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = TensorRTUniquePtr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(explicit_batch));
    if (!network) {
        cout << "error 2" << endl;

        return false;
    }

    auto config = TensorRTUniquePtr<nvinfer1::IBuilderConfig>(
        builder->createBuilderConfig());
    if (!config) {
        cout << "error 3" << endl;

        return false;
    }

    auto parser = TensorRTUniquePtr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, gLogger.getTRTLogger()));
    if (!parser) {
        cout << "error 4" << endl;

        return false;
    }

    auto profile = builder->createOptimizationProfile();
    if (!profile) {
        cout << "error 5" << endl;

        return false;
    }

    // .engine输入点不在这范围内时，将导致维度检验失败而core
    constexpr int kMinPointNum = SuperPointConfig::kMaxKeypoints;
    constexpr int kMidPointNum = SuperPointConfig::kMaxKeypoints;
    constexpr int kMaxPointNum = SuperPointConfig::kMaxKeypoints;
    // 两对输入特征点
    profile->setDimensions(LightGlueConfig::kInputTensorNames[0],
                           OptProfileSelector::kMIN, Dims3(1, kMinPointNum, 2));
    profile->setDimensions(LightGlueConfig::kInputTensorNames[0],
                           OptProfileSelector::kOPT, Dims3(1, kMidPointNum, 2));
    profile->setDimensions(LightGlueConfig::kInputTensorNames[0],
                           OptProfileSelector::kMAX, Dims3(1, kMaxPointNum, 2));

    profile->setDimensions(LightGlueConfig::kInputTensorNames[1],
                           OptProfileSelector::kMIN, Dims3(1, kMinPointNum, 2));
    profile->setDimensions(LightGlueConfig::kInputTensorNames[1],
                           OptProfileSelector::kOPT, Dims3(1, kMidPointNum, 2));
    profile->setDimensions(LightGlueConfig::kInputTensorNames[1],
                           OptProfileSelector::kMAX, Dims3(1, kMaxPointNum, 2));
    // 两对输入描述子
    profile->setDimensions(LightGlueConfig::kInputTensorNames[2],
                           OptProfileSelector::kMIN,
                           Dims3(1, kMinPointNum, 256));
    profile->setDimensions(LightGlueConfig::kInputTensorNames[2],
                           OptProfileSelector::kOPT,
                           Dims3(1, kMidPointNum, 256));
    profile->setDimensions(LightGlueConfig::kInputTensorNames[2],
                           OptProfileSelector::kMAX,
                           Dims3(1, kMaxPointNum, 256));

    profile->setDimensions(LightGlueConfig::kInputTensorNames[3],
                           OptProfileSelector::kMIN,
                           Dims3(1, kMinPointNum, 256));
    profile->setDimensions(LightGlueConfig::kInputTensorNames[3],
                           OptProfileSelector::kOPT,
                           Dims3(1, kMidPointNum, 256));
    profile->setDimensions(LightGlueConfig::kInputTensorNames[3],
                           OptProfileSelector::kMAX,
                           Dims3(1, kMaxPointNum, 256));

    // 输出不用设置
    config->addOptimizationProfile(profile);
    auto constructed = ConstructNetwork(builder, network, config, parser);
    if (!constructed) {
        cout << "error 6" << endl;

        return false;
    }

    auto profile_stream = makeCudaStream();
    if (!profile_stream) {
        cout << "error 7" << endl;

        return false;
    }
    config->setProfileStream(*profile_stream);

    TensorRTUniquePtr<IHostMemory> plan{
        builder->buildSerializedNetwork(*network, *config)};
    if (!plan) {
        cout << "error 8" << endl;

        return false;
    }

    TensorRTUniquePtr<IRuntime> runtime{
        createInferRuntime(gLogger.getTRTLogger())};
    if (!runtime) {
        cout << "error 9" << endl;

        return false;
    }

    engine_ = shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()));
    if (!engine_) {
        cout << "error 10" << endl;

        return false;
    }

    SaveEngine();

    ASSERT(network->getNbInputs() == 4);
    // 特征点坐标的输入维度
    keypoints_0_dims_ = network->getInput(0)->getDimensions();
    keypoints_1_dims_ = network->getInput(1)->getDimensions();

    // 特征点对应描述子的维度
    descriptors_0_dims_ = network->getInput(2)->getDimensions();
    descriptors_1_dims_ = network->getInput(3)->getDimensions();

    assert(keypoints_0_dims_.d[1] == -1);  // 特征点数量维度不确定
    assert(keypoints_1_dims_.d[1] == -1);

    assert(descriptors_0_dims_.d[1] == -1);  // 描述子数量维度不确定
    assert(descriptors_1_dims_.d[1] == -1);
    return true;
}

bool LightGlue::ConstructNetwork(
    TensorRTUniquePtr<nvinfer1::IBuilder>& builder,
    TensorRTUniquePtr<nvinfer1::INetworkDefinition>& network,
    TensorRTUniquePtr<nvinfer1::IBuilderConfig>& config,
    TensorRTUniquePtr<nvonnxparser::IParser>& parser) const {

    if (!parser->parseFromFile(
            onnxFilePath_.c_str(),
            static_cast<int>(nvinfer1::ILogger::Severity::kVERBOSE))) {
        cerr << "ONNX parse failed.\n";
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            auto* e = parser->getError(i);
            cerr << "Parser error " << i << ": " << e->desc() << "\n";
        }
        return false;
    }

    config->setMaxWorkspaceSize(512_MiB);
    config->setFlag(BuilderFlag::kFP16);  // 注释掉即使用FP32测试
    return true;
}

bool LightGlue::Infer(
    const Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor>& kpts0,
    const Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor>& kpts1,
    const Eigen::Matrix<float, Eigen::Dynamic, 256, Eigen::RowMajor>& desc0,
    const Eigen::Matrix<float, Eigen::Dynamic, 256, Eigen::RowMajor>& desc1,
    Eigen::VectorXi& indices0, Eigen::VectorXi& indices1,
    Eigen::VectorXf& mscores) {
    if (!context_) {
        context_ = TensorRTUniquePtr<nvinfer1::IExecutionContext>(
            engine_->createExecutionContext());
        if (!context_) {
            cerr << "lightglue infer error 1" << endl;
            return false;
        }
    }

    assert(engine_->getNbBindings() == 5);

    // 输入特征点索引
    const int keypoints_0_index =
        engine_->getBindingIndex(LightGlueConfig::kInputTensorNames[0]);
    const int keypoints_1_index =
        engine_->getBindingIndex(LightGlueConfig::kInputTensorNames[1]);

    // 输入描述子索引
    const int descriptors_0_index =
        engine_->getBindingIndex(LightGlueConfig::kInputTensorNames[2]);
    const int descriptors_1_index =
        engine_->getBindingIndex(LightGlueConfig::kInputTensorNames[3]);

    // 输出得分索引，得分与描述子内积有关
    const int output_scores_index =
        engine_->getBindingIndex(LightGlueConfig::kOutputTensorNames[0]);

    // 设置输入特征点数量，用于分配输入向量内存
    // 注意，我们这里使用了Eigen的行优先存储规则
    context_->setBindingDimensions(keypoints_0_index,
                                   Dims3(1, kpts0.rows(), 2));
    context_->setBindingDimensions(keypoints_1_index,
                                   Dims3(1, kpts1.rows(), 2));

    context_->setBindingDimensions(descriptors_0_index,
                                   Dims3(1, desc0.rows(), 256));
    context_->setBindingDimensions(descriptors_1_index,
                                   Dims3(1, desc1.rows(), 256));

    keypoints_0_dims_ = context_->getBindingDimensions(keypoints_0_index);
    keypoints_1_dims_ = context_->getBindingDimensions(keypoints_1_index);

    descriptors_0_dims_ = context_->getBindingDimensions(descriptors_0_index);
    descriptors_1_dims_ = context_->getBindingDimensions(descriptors_1_index);

    // 关键修复：检查所有输入维度是否已指定
    if (!context_->allInputDimensionsSpecified()) {
        cerr << "Error: Not all input dimensions are specified!" << endl;
        return false;
    }

    // 要对输出也设置绑定
    output_scores_dims_ = context_->getBindingDimensions(output_scores_index);

    // 输出看一下tensorrt是否能够从输入推断出输出的类型，
    // 只有能够推断才能正确运行，输出为：
    /*
    Input 0 name=kpts0 dtype=0 dims=[1,341,2]
    Input 1 name=kpts1 dtype=0 dims=[1,350,2]
    Input 2 name=desc0 dtype=0 dims=[1,341,256]
    Input 3 name=desc1 dtype=0 dims=[1,350,256]

    Output 4 name=scores dtype=0 dims=[1,341,350]
    Output 5 name=matchability0 dtype=0 dims=[1,341,0] # 0表示该维度没有意义
    Output 6 name=matchability1 dtype=0 dims=[1,350,0]
    kpts0 num: 341, kpts1 num: 350, desc0 num: 341, desc1 num: 350
*/
    // for (int i = 0; i < engine_->getNbBindings(); ++i) {
    //     auto t = engine_->getBindingDataType(i);
    //     auto dims = context_->getBindingDimensions(i);
    //     cout << (engine_->bindingIsInput(i) ? "Input " : "Output ") << i
    //               << " name=" << engine_->getBindingName(i)
    //               << " dtype=" << static_cast<int>(t) << " dims=[" << dims.d[0]
    //               << "," << dims.d[1] << "," << dims.d[2] << "]\n";
    // }

    BufferManager buffers(engine_, 0, context_.get());

    if (!ProcessInput(buffers, kpts0, kpts1, desc0, desc1)) {
        cout << "lightglue infer error 2" << endl;
        return false;
    }

    cout << "kpts0 num: " << kpts0.rows() << ", kpts1 num: " << kpts1.rows()
         << ", desc0 num: " << desc0.rows() << ", desc1 num: " << desc1.rows()
         << endl;

    buffers.copyInputToDevice();

    auto start = chrono::high_resolution_clock::now();
    bool status = context_->executeV2(buffers.getDeviceBindings().data());
    if (!status) {
        cout << "lightglue infer error 3" << endl;
        return false;
    }
    buffers.copyOutputToHost();
    auto end = chrono::high_resolution_clock::now();
    const double spendTime =
        chrono::duration_cast<chrono::milliseconds>(end - start).count();
    cout << fmt::format("lightglue executeV2 spend: {:.1f}ms", spendTime)
         << endl;

    if (!ProcessOutput(buffers, indices0, indices1, mscores)) {
        cout << "lightglue infer error 4" << endl;

        return false;
    }

    return true;
}

bool LightGlue::ProcessInput(
    const BufferManager& buffers,
    const Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor>& kpts0,
    const Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor>& kpts1,
    const Eigen::Matrix<float, Eigen::Dynamic, 256, Eigen::RowMajor>& desc0,
    const Eigen::Matrix<float, Eigen::Dynamic, 256, Eigen::RowMajor>& desc1) {

#if !USE_FP16_PRECISION
    auto* keypoints_0_buffer = static_cast<float*>(
        buffers.getHostBuffer(LightGlueConfig::kInputTensorNames[0]));
    auto* keypoints_1_buffer = static_cast<float*>(
        buffers.getHostBuffer(LightGlueConfig::kInputTensorNames[1]));

    auto* descriptors_0_buffer = static_cast<float*>(
        buffers.getHostBuffer(LightGlueConfig::kInputTensorNames[2]));
    auto* descriptors_1_buffer = static_cast<float*>(
        buffers.getHostBuffer(LightGlueConfig::kInputTensorNames[3]));

    // 给输入特征点向量buffer赋值，TODO：注意输入坐标系的顺序，
    // 需注意网络要求的是float16还是float32，只在float32输入时可用memcpy
    memcpy(keypoints_0_buffer, kpts0.data(), kpts0.size() * sizeof(float));
    memcpy(keypoints_1_buffer, kpts1.data(), kpts1.size() * sizeof(float));

    // 给输入描述子向量buffer赋值，TODO：需要注意输入顺序
    memcpy(descriptors_0_buffer, desc0.data(), desc0.size() * sizeof(float));
    memcpy(descriptors_1_buffer, desc1.data(), desc1.size() * sizeof(float));

#else
    auto* keypoints_0_buffer = static_cast<__half*>(
        buffers.getHostBuffer(LightGlueConfig::kInputTensorNames[0]));
    auto* keypoints_1_buffer = static_cast<__half*>(
        buffers.getHostBuffer(LightGlueConfig::kInputTensorNames[1]));

    auto* descriptors_0_buffer = static_cast<__half*>(
        buffers.getHostBuffer(LightGlueConfig::kInputTensorNames[2]));
    auto* descriptors_1_buffer = static_cast<__half*>(
        buffers.getHostBuffer(LightGlueConfig::kInputTensorNames[3]));

    // 给输入特征点向量buffer赋值，TODO：注意输入坐标系的顺序，
    // 需注意网络要求的是float16还是float32，只在float32输入时可用memcpy
    for (int i = 0; i < kpts0.rows(); ++i) {
        keypoints_0_buffer[i * 2 + 0] = static_cast<__half>(kpts0(i, 0));
        keypoints_0_buffer[i * 2 + 1] = static_cast<__half>(kpts0(i, 1));
        for (int j = 0; j < 256; ++j) {
            descriptors_0_buffer[256 * i + j] =
                static_cast<__half>(desc0(i, j));
        }
    }

    for (int i = 0; i < kpts1.rows(); ++i) {
        keypoints_1_buffer[i * 2 + 0] = static_cast<__half>(kpts1(i, 0));
        keypoints_1_buffer[i * 2 + 1] = static_cast<__half>(kpts1(i, 1));
        for (int j = 0; j < 256; ++j) {
            descriptors_1_buffer[256 * i + j] =
                static_cast<__half>(desc1(i, j));
        }
    }
#endif
    return true;
}

bool LightGlue::ProcessOutput(const BufferManager& buffers,
                              Eigen::VectorXi& indices0,
                              Eigen::VectorXi& indices1,
                              Eigen::VectorXf& mscores) {
    indices0_.clear();
    indices1_.clear();
    mscores_.clear();

#if USE_FP16_PRECISION
    auto* output_scores_half = static_cast<__half*>(
        buffers.getHostBuffer(LightGlueConfig::kOutputTensorNames[0]));
    vector<float> vecOutputScore(output_scores_dims_.d[1] *
                                 output_scores_dims_.d[2]);
    for (size_t i = 0; i < vecOutputScore.size(); ++i) {
        vecOutputScore[i] = exp(static_cast<float>(output_scores_half[i]));
    }
    float* output_scores = vecOutputScore.data();
#else
    // 获取匹配得分数组
    auto* output_scores = static_cast<float*>(
        buffers.getHostBuffer(LightGlueConfig::kOutputTensorNames[0]));
    // cout << "Lightglue output score Dim: " << output_scores_dims_.nbDims
    //           << ", d[0]: " << output_scores_dims_.d[0] << endl;
    if (output_scores_dims_.d[0] < 1) {
        return false;
    }

    // Eigen::MatrixXf debugInferResult;
    // debugInferResult.resize(output_scores_dims_.d[1], output_scores_dims_.d[2]);
    // debugInferResult.setConstant(-1.11);

    // scores通过exp转为概率
    int scoreId = 0;
    for (int row = 0; row < output_scores_dims_.d[1]; ++row) {
        for (int col = 0; col < output_scores_dims_.d[2]; ++col) {
            output_scores[scoreId] = exp(output_scores[scoreId]);
            ++scoreId;
            // debugInferResult(row, col) = output_scores[scoreId];
        }
    }
    // cout << "debugInferResult:\n" << debugInferResult << endl;
#endif

    // 匹配得分矩阵后处理
    vector<MatchPair> matches;
    ExtractMatches(output_scores, matches, keypoints_0_dims_.d[1],
                   keypoints_1_dims_.d[1]);
    cout << "Lightglue find matches num: " << matches.size() << endl;

    indices0.resize(keypoints_0_dims_.d[1]);
    indices1.resize(keypoints_1_dims_.d[1]);
    indices0.setConstant(-1);
    indices1.setConstant(-1);
    mscores.resize(keypoints_0_dims_.d[1]);  // 得分以第一张图像的关键点顺序记录
    for (size_t i = 0; i < matches.size(); ++i) {
        const int first_img_kp_idx = matches[i].index0;
        const int second_img_kp_idx = matches[i].index1;
        indices0[first_img_kp_idx] = second_img_kp_idx;
        indices1[second_img_kp_idx] = first_img_kp_idx;
        mscores[first_img_kp_idx] = matches[i].score;
    }

    return true;
}

void LightGlue::SaveEngine() {
    if (engineFilePath_.empty())
        return;
    if (engine_ != nullptr) {
        nvinfer1::IHostMemory* data = engine_->serialize();
        ofstream file(engineFilePath_, ios::binary);
        ;
        if (!file)
            return;
        file.write(reinterpret_cast<const char*>(data->data()), data->size());
    }
}

bool LightGlue::DeserializeEngine() {
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

int LightGlue::MatchKeypoints(
    Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor>& kpts0,
    Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor>& kpts1,
    Eigen::Matrix<float, Eigen::Dynamic, 256, Eigen::RowMajor>& desc0,
    Eigen::Matrix<float, Eigen::Dynamic, 256, Eigen::RowMajor>& desc1,
    Eigen::VectorXf& mscores, vector<cv::DMatch>& matches,
    bool outlier_rejection) {
    matches.clear();

    // 归一化坐标点
    Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor> normKpts0 =
        NormalizeKeypoints(kpts0, LightGlueConfig::kImageWidth,
                           LightGlueConfig::kImageHeight);
    Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor> normKpts1 =
        NormalizeKeypoints(kpts1, LightGlueConfig::kImageWidth,
                           LightGlueConfig::kImageHeight);
    /*
    string kpts0NormMessage("kpts0 = np.array([[");
    for (int i = 0; i < normKpts0.rows(); ++i) {
        const Eigen::Vector2f p = normKpts0.row(i);
        if (i == 0) {
            kpts0NormMessage.append(
                fmt::format("[{:.6f}, {:.6f}]", p.x(), p.y()));
        } else {
            kpts0NormMessage.append(
                fmt::format(", [{:.6f}, {:.6f}]", p.x(), p.y()));
        }
    }
    kpts0NormMessage.append("]]).astype(np.float32)");

    string kpts1NormMessage("kpts1 = np.array([[");
    for (int i = 0; i < normKpts0.rows(); ++i) {
        const Eigen::Vector2f p = normKpts1.row(i);
        if (i == 0) {
            kpts1NormMessage.append(
                fmt::format("[{:.6f}, {:.6f}]", p.x(), p.y()));
        } else {
            kpts1NormMessage.append(
                fmt::format(", [{:.6f}, {:.6f}]", p.x(), p.y()));
        }
    }
    kpts1NormMessage.append("]]).astype(np.float32)");

    auto GetDescriptorStr =
        [](const Eigen::Matrix<float, 1, 256, Eigen::RowMajor>& desc)
        -> string {
        string s("[");
        for (int i = 0; i < 256; ++i) {
            if (i == 0) {
                s.append(fmt::format("{}", desc[i]));
            } else {
                s.append(fmt::format(", {} ", desc[i]));
            }
        }
        s.append("]");
        return s;
    };

    string desc0NormMessage("desc0 = np.array([[");
    for (int i = 0; i < desc0.rows(); ++i) {
        // desc0NormMessage.append(fmt::format("{} ", desc0.row(i).norm()));
        if (i == 0) {
            desc0NormMessage.append(GetDescriptorStr(desc0.row(i)));
        } else {
            desc0NormMessage.append(", ");
            desc0NormMessage.append(GetDescriptorStr(desc0.row(i)));
        }
    }
    desc0NormMessage.append("]]).astype(np.float32)");

    string desc1NormMessage("desc1 = np.array([[");
    for (int i = 0; i < desc1.rows(); ++i) {
        // desc1NormMessage.append(fmt::format("{} ", desc1.row(i).norm()));
        if (i == 0) {
            desc1NormMessage.append(GetDescriptorStr(desc1.row(i)));
        } else {
            desc1NormMessage.append(", ");
            desc1NormMessage.append(GetDescriptorStr(desc1.row(i)));
        }
    }
    desc1NormMessage.append("]]).astype(np.float32)");

    cout << kpts0NormMessage << endl
              << desc0NormMessage << endl
              << kpts1NormMessage << endl
              << desc1NormMessage << endl;

    cout << "kpts0:\n"
              << kpts0 << "\nkpts1:\n"
              << kpts1 << "\ndesc0:\n"
              << desc0 << "\ndesc1:\n"
              << desc1 << endl;
*/

    // 模型推理
    Eigen::VectorXi indices0, indices1;
    Infer(normKpts0, normKpts1, desc0, desc1, indices0, indices1, mscores);

    int num_match = 0;
    vector<cv::Point2f> points0, points1;
    vector<int> point_indexes;
    for (int i = 0; i < indices0.rows(); i++) {
        if (indices0(i) < indices1.size() && indices0(i) >= 0 &&
            indices1(indices0(i)) == i) {
            const float d = mscores(i);
            matches.emplace_back(i, indices0[i], d);
            points0.emplace_back(kpts0(i, 0), kpts0(i, 1));
            points1.emplace_back(kpts1(indices0(i), 0), kpts1(indices0(i), 1));
            num_match++;
        }
    }

    if (outlier_rejection) {
        vector<uchar> inliers;
        cv::findFundamentalMat(points0, points1, cv::FM_RANSAC, 3, 0.999,
                               inliers);
        int j = 0;
        // 内存复用
        for (int i = 0; i < static_cast<int>(matches.size()); i++) {
            if (inliers[i]) {
                matches[j++] = matches[i];
            }
        }
        // 仅保留内点匹配j个
        matches.resize(j);
    }

    return matches.size();
}

Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor>
LightGlue::NormalizeKeypoints(
    const Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor>& kpts,
    int width, int height) {
    Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor> norm_kpts = kpts;

    const float cx = 0.5f * static_cast<float>(width);
    const float cy = 0.5f * static_cast<float>(height);
    const float scale = 0.5f * static_cast<float>(max(width, height));

    for (int row = 0; row < kpts.rows(); ++row) {
        norm_kpts(row, 0) = (kpts(row, 0) - cx) / scale;
        norm_kpts(row, 1) = (kpts(row, 1) - cy) / scale;
    }
    return norm_kpts;
}

void LightGlue::ExtractMatches(const float* scores, vector<MatchPair>& matches,
                               int num_kpts0, int num_kpts1) {
    matches.clear();

    // 计算互最近邻
    vector<int> matches0(num_kpts0, -1);
    vector<int> matches1(num_kpts1, -1);
    vector<float> scores0(num_kpts0, 0.0f);
    vector<float> scores1(num_kpts1, 0.0f);

    MutualNearestNeighbor(scores, matches0, matches1, scores0, scores1,
                          num_kpts0, num_kpts1);

    // 应用阈值过滤
    ApplyThreshold(matches0, matches1, scores0, scores1, threshold_);

    // 收集有效匹配
    for (int i = 0; i < num_kpts0; ++i) {
        if (matches0[i] != -1) {
            int j = matches0[i];
            // 结合匹配性得分
            if (scores0[i] > threshold_) {
                matches.emplace_back(i, j, scores0[i]);
            }
        }
    }

    // 按得分排序
    sort(matches.begin(), matches.end(),
         [](const MatchPair& a, const MatchPair& b) {
             return a.score > b.score;
         });
}

void LightGlue::ExtractMatchesFast(const float* scores,
                                   vector<MatchPair>& matches, int num_kpts0,
                                   int num_kpts1) {
    matches.clear();

    vector<int> matches0(num_kpts0, -1);
    vector<int> matches1(num_kpts1, -1);
    vector<float> scores0(num_kpts0, 0.0f);
    vector<float> scores1(num_kpts1, 0.0f);

    MutualNearestNeighbor(scores, matches0, matches1, scores0, scores1,
                          num_kpts0, num_kpts1);
    ApplyThreshold(matches0, matches1, scores0, scores1, threshold_);

    for (int i = 0; i < num_kpts0; ++i) {
        if (matches0[i] != -1) {
            matches.emplace_back(i, matches0[i], scores0[i]);
        }
    }
}

void LightGlue::MutualNearestNeighbor(const float* scores,
                                      vector<int>& matches0,
                                      vector<int>& matches1,
                                      vector<float>& scores0,
                                      vector<float>& scores1, int num_kpts0,
                                      int num_kpts1) {
    // 为关键点0找到最佳匹配
    for (int i = 0; i < num_kpts0; ++i) {
        float best_score = -numeric_limits<float>::max();
        int best_j = -1;

        for (int j = 0; j < num_kpts1; ++j) {
            float score = scores[i * num_kpts1 + j];
            if (score > best_score) {
                best_score = score;
                best_j = j;
            }
        }

        if (best_j != -1) {
            matches0[i] = best_j;
            scores0[i] = best_score;  // 转换回概率
        }

        // cout << "kpts0[" << i << "]<=>kpts1[" << best_j
        //           << "], best score: " << scores0[i] << endl;
    }

    // 为关键点1找到最佳匹配
    for (int j = 0; j < num_kpts1; ++j) {
        float best_score = -numeric_limits<float>::max();
        int best_i = -1;

        for (int i = 0; i < num_kpts0; ++i) {
            float score = scores[i * num_kpts1 + j];
            if (score > best_score) {
                best_score = score;
                best_i = i;
            }
        }

        if (best_i != -1) {
            matches1[j] = best_i;
            scores1[j] = best_score;
        }

        // cout << "kpts1[" << j << "]<=>kpts0[" << best_i
        //           << "], best score: " << scores1[j] << endl;
    }

    // 检查互最近邻
    for (int i = 0; i < num_kpts0; ++i) {
        if (matches0[i] != -1) {
            int j = matches0[i];
            if (matches1[j] != i) {
                matches0[i] = -1;  // 不是互最近邻
                scores0[i] = 0.0f;
            }
        }
    }
}

void LightGlue::ApplyThreshold(vector<int>& matches0, vector<int>& matches1,
                               vector<float>& scores0, vector<float>& scores1,
                               float threshold) {
    for (size_t i = 0; i < matches0.size(); ++i) {
        if (matches0[i] != -1 && scores0[i] < threshold) {
            matches0[i] = -1;
            scores0[i] = 0.0f;
        }
    }

    for (size_t j = 0; j < matches1.size(); ++j) {
        if (matches1[j] != -1 && scores1[j] < threshold) {
            matches1[j] = -1;
            scores1[j] = 0.0f;
        }
    }
}

bool LightGlue::ValidateFP16() {
    if (!engine_) {
        std::cout << "引擎未构建" << std::endl;
        return false;
    }

    std::cout << "\n=== FP16启用状态验证 ===" << std::endl;

    // 方法1: 检查绑定数据类型（推荐，兼容所有版本）
    int fp16_bindings = 0;
    int total_bindings = engine_->getNbBindings();

    for (int i = 0; i < total_bindings; ++i) {
        auto dtype = engine_->getBindingDataType(i);
        const char* name = engine_->getBindingName(i);

        std::string dtype_str;
        switch (dtype) {
            case nvinfer1::DataType::kFLOAT:
                dtype_str = "FP32";
                break;
            case nvinfer1::DataType::kHALF:
                dtype_str = "FP16";
                fp16_bindings++;
                break;
            case nvinfer1::DataType::kINT8:
                dtype_str = "INT8";
                break;
            default:
                dtype_str = "UNKNOWN";
        }

        std::cout << "绑定 " << i << " (" << name << "): " << dtype_str
                  << (engine_->bindingIsInput(i) ? " [输入]" : " [输出]")
                  << std::endl;
    }

    // 方法2: 检查引擎的FP16标志（如果有）
    bool fp16_enabled = false;

    // 检查是否有任何绑定使用FP16
    if (fp16_bindings > 0) {
        fp16_enabled = true;
        std::cout << "✅ FP16已成功启用 (" << fp16_bindings << "/"
                  << total_bindings << " 个绑定使用FP16)" << std::endl;
    } else {
        std::cout << "❌ FP16未启用，所有绑定使用FP32" << std::endl;
    }

    // 方法3: 检查性能特征（间接验证）
    CheckPerformanceCharacteristics();

    return fp16_enabled;
}

void LightGlue::CheckPerformanceCharacteristics() {
    std::cout << "\n=== 性能特征分析 ===" << std::endl;

    // 估算内存使用
    size_t total_memory = 0;
    for (int i = 0; i < engine_->getNbBindings(); ++i) {
        auto dims = engine_->getBindingDimensions(i);
        auto dtype = engine_->getBindingDataType(i);

        size_t element_size = 0;
        switch (dtype) {
            case nvinfer1::DataType::kFLOAT:
                element_size = 4;
                break;
            case nvinfer1::DataType::kHALF:
                element_size = 2;
                break;
            case nvinfer1::DataType::kINT8:
                element_size = 1;
                break;
            default:
                element_size = 4;
        }

        // 估算张量大小（使用优化配置的最大维度）
        int64_t element_count = 1;
        for (int j = 0; j < dims.nbDims; ++j) {
            if (dims.d[j] == -1) {
                // 动态维度，使用最大值估算
                element_count *= 1024;  // 使用配置的最大值
            } else {
                element_count *= dims.d[j];
            }
        }

        total_memory += element_count * element_size;
    }

    std::cout << "估算内存占用: " << total_memory / (1024 * 1024) << " MB"
              << std::endl;

    if (total_memory < 100 * 1024 * 1024) {  // 小于100MB
        std::cout << "内存占用较低，可能使用了FP16量化" << std::endl;
    } else {
        std::cout << "内存占用较高，可能使用FP32" << std::endl;
    }
}