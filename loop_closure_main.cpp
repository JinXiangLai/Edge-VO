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

#define MANUAL_SCALE_POSITION 1

#define DEBUG_POSE_GRAPH_SE3 0

#define USE_SPARSE_H_MATRIX 1

constexpr double kScaleDriftRatio = 0.1;

const char* const kBeforeLoopClosurePoseFilePath =
    "./before_loop_closure_pose.txt";
const char* const kClosurePoseFilePath = "./loop_closure_pose.txt";

constexpr double kW[3] = {1.0, 1.0, 1.0};  // ΔR, ΔP, Δs对应的权重

void Run(Optimizer* optimizer);

const cv::Point kViz3DWindowPos(1920 + 1920 / 2,
                                1080 - 100);  // 窗口左上角点在屏幕上的位置

void ResetStatus(Optimizer* optimizer, bool* isInitialized,
                 int* trackLostCount);

bool RecoveryPose(const shared_ptr<Camera> cam, const KeyFrame& lastKf,
                  const KeyFrame& curF, const vector<cv::Point2f>& pts1,
                  const vector<cv::Point2f>& pts2, Pose& Tw2);

bool Sim3PoseGraphOptimization(vector<KeyFrame*>& allKeyframe, int fixedIndex,
                               int loopClosureIndex);

bool CalculateRelativeSim3Transform(const KeyFrame* fixedKF,
                                    const KeyFrame* loopClosureKF,
                                    Sim3Pose& relativeSim3T12);

bool UmeyamaSim3Transform(const vector<Eigen::Vector3d>& ps1,
                          const vector<Eigen::Vector3d>& ps2, Sim3Pose& T12);

int CalculateLoopClosureSim3PoseAndConstraint(
    const Sim3Pose& relativeSim3T12, const vector<KeyFrame*>& selectKFresult,
    vector<Sim3Pose>& loopClosurePoseTwc,
    vector<Sim3Pose>& relativePoseConstraint);

// 需保证首帧为参考帧，未帧为回环帧
void ConstructPoseGraphHessianandGradiant(
    const vector<Sim3Pose>& sTwc, const vector<Sim3Pose>& sT12Constraint,
    Eigen::MatrixXd& H, Eigen::VectorXd& g);
// 使用稀疏信息矩阵H
void ConstructPoseGraphSparseHessianandGradiant(
    const vector<Sim3Pose>& sTwc, const vector<Sim3Pose>& sT12Constraint,
    vector<Eigen::Triplet<double>>& triplets, Eigen::SparseMatrix<double>& H,
    Eigen::VectorXd& g);

bool SelectKeyframeInLoopClosure(vector<KeyFrame*>& allKeyframe, int fixedIndex,
                                 int loopClosureIndex,
                                 vector<KeyFrame*>& selectResult);

Optimizer::ResidualInfo CalculatePoseGraphResidualInfo(
    const vector<Sim3Pose>& sTwc, const vector<Sim3Pose>& sT12Constraint);

Eigen::Matrix<double, 7, 1> Sim3PoseResidual(const Sim3Pose& sT12,
                                             const Sim3Pose& sTprior12);

double RelativePoseHuberLoss(const Eigen::Matrix<double, 7, 1>& residual);

double ComputePredictionReduction(const double lambda,
                                  const Eigen::VectorXd& deltaX,
                                  const Eigen::VectorXd& g,
                                  const Eigen::MatrixXd& H);

double ComputePredictionReductionSparseHessian(
    const double lambda, const Eigen::VectorXd& deltaX,
    const Eigen::VectorXd& g, const Eigen::SparseMatrix<double>& H);

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

    constexpr int kStep = 10;
    double accumulateMoveDist = 0.;
    for (size_t i = firstImgIdx; i < vTimeStamps.size(); i += kStep) {
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

#if !MANUAL_SCALE_POSITION
            // 初始化匹配器
            lightglue = KeyFrame::lightgluePtr;
            lightglue->SetThreshold(0.05);
            curKF.ExtractSuperpoint();
#endif
            const cv::Mat& gray = curKF.grayImg_;

            matchImage = cv::Mat(gray.rows, gray.cols * 2, CV_8UC1);
            matchImgColor = cv::Mat(gray.rows, gray.cols * 2, CV_8UC3);
            gray.copyTo(matchImage(cv::Rect(0, 0, gray.cols, gray.rows)));
            if (!videoSavePath.empty()) {
#if CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR < 6
                writer = cv::cudacodec::createVideoWriter(
                    videoSavePath, matchImgColor.size(), 24.0);
#else
                writer = cv::cudacodec::createVideoWriter(videoSavePath,
                                                          matchImgColor.size());
#endif
            }

            KeyFrame::InitPoseFileMessage();
            kfVec.emplace_back(curKF);
            continue;
        }

        const double moveDist = (curKF.Tcw_ * curF.Twc_).t_wb_.norm();
        accumulateMoveDist += moveDist;
        if (moveDist < 0.08) {
            continue;
        }

#if !MANUAL_SCALE_POSITION
        curF.ExtractSuperpoint();
#endif
        const cv::Mat& gray = curF.grayImg_;
        gray.copyTo(matchImage(cv::Rect(gray.cols, 0, gray.cols, gray.rows)));

        Eigen::VectorXf mscores;
        vector<cv::DMatch> lightglueMatches;
#if !MANUAL_SCALE_POSITION
        const int matchPairNum =
            lightglue->MatchKeypoints(curKF.kpts_, curF.kpts_, curKF.desc_,
                                      curF.desc_, mscores, lightglueMatches);
#else
        const int matchPairNum = 0;
#endif
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
            ++count;
        }

        // 获取匹配点，使用本质矩阵分解及先验尺度计算每个帧的pose
        Pose Twc2;
        const bool getPoseStatus =
            RecoveryPose(cam, curKF, curF, pts0, pts1, Twc2);
        if (getPoseStatus) {
            curF.SetTwc(Twc2);
        } else {
            cout << fmt::format("Error while recovery Twc2, match pairs: {}\n",
                                pts0.size());
        }

        cv::putText(matchImgColor,
                    fmt::format("match num: {}, recovery pose: {}",
                                matchPairNum, getPoseStatus),
                    cv::Point(10, 30), cv::FONT_ITALIC, 0.80, {0, 0, 255}, 2);

        bool needNewKf =
            getPoseStatus && (matchPairNum < 0.7 * pts0.size() ||
                              (curF.timestamp_ - curKF.timestamp_) > 2.0);
#if MANUAL_SCALE_POSITION
        needNewKf = getPoseStatus;
#endif
        if (needNewKf) {
            cout << fmt::format("Select {}th img as image0", curF.id_) << endl;
            curKF = std::move(curF);
            gray.copyTo(matchImage(cv::Rect(0, 0, gray.cols, gray.rows)));

            kfVec.emplace_back(curKF);

            const int loopClosureIndex = kfVec.size() - 1;
            int fixedIndex = -1;
            double loopClosureAccDist = 0.;
            if (kfVec.size() > 5 && accumulateMoveDist > 1.5) {

                for (int i = kfVec.size() - 2; i >= 0; --i) {
                    const double moveDist =
                        (kfVec[i].priorTwc_.Inverse() * kfVec[i + 1].priorTwc_)
                            .t_wb_.norm();
                    loopClosureAccDist += moveDist;
                    if (loopClosureIndex - i < 10 || loopClosureAccDist < 1.0) {
                        continue;
                    }

                    const Pose T12 =
                        kfVec[i].priorTwc_.Inverse() * curKF.priorTwc_;
                    const double dpNorm = T12.t_wb_.norm();
                    const bool exeLoopClosure = loopClosureIndex - i > 10 &&
                                                dpNorm < 0.2 &&
                                                loopClosureAccDist > 2.0;
                    // dpNorm > 0.03
                    if (!exeLoopClosure) {
                        continue;
                    }

                    fixedIndex = i;
                    break;
                }

                if (fixedIndex >= 0) {
                    vector<KeyFrame*> allKeyframe;
                    for (auto& kf : kfVec) {
                        allKeyframe.emplace_back(&kf);
                    }
                    cout << fmt::format("Find loop closure id range: [{}, {}]",
                                        fixedIndex, loopClosureIndex)
                         << endl;
                    Sim3PoseGraphOptimization(allKeyframe, fixedIndex,
                                              loopClosureIndex);
                    break;
                }
            }
        }

        if (!videoSavePath.empty() && needNewKf) {
            gpuFrame.upload(matchImgColor);
            writer->write(gpuFrame);
        }
    }

    for (const KeyFrame& f : kfVec) {
        KeyFrame::WritePoseMessage2File(f);
    }
    KeyFrame::ProcessPoseFile();

    // 没有产生Landmar*，因此比较poseFile
    cout << "Run evo evaluate kf traj precision!";
    system(fmt::format("evo_ape tum {}/groundtruth.txt {} -a -s -v",
                       config->dataDir, KeyFrame::poseFilePath)
               .c_str());
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
    globalOptFlw.Reset();
    *trackLostCount = 0;
    interaction->trajectory.clear();
}

bool RecoveryPose(const shared_ptr<Camera> cam, const KeyFrame& lastKf,
                  const KeyFrame& curF, const vector<cv::Point2f>& pts1,
                  const vector<cv::Point2f>& pts2, Pose& Tw2) {
#if !MANUAL_SCALE_POSITION
    // opencv输出是T21，因此本质矩阵输入要倒序
    cv::Mat cvE =
        cv::findEssentialMat(pts2, pts1, Eigen2CVmat(cam->K_[0]), cv::RANSAC);

    // 2. 分解本质矩阵得到R,t，
    cv::Mat cvR, cv_t, mask;
    int inliers =
        recoverPose(cvE, pts2, pts1, Eigen2CVmat(cam->K_[0]), cvR, cv_t, mask);
    cout << "OpenCV recoverPose found " << inliers << " inliers" << endl;
    cout << "cvR: " << cvR << endl;
    cout << "cv_t: " << cv_t.t() << ", norm: " << cv::norm(cv_t) << endl;

    const Eigen::Matrix3d Rres = CVmat2Eigen(cvR);
    Eigen::Quaterniond q(Rres);
    Pose initT12(q, CVmat2Eigen(cv_t));

    Pose trueT12 = lastKf.priorTwc_.Inverse() * curF.priorTwc_;
    const double scale = trueT12.t_wb_.norm() / initT12.t_wb_.norm();
    initT12.t_wb_ *= scale;
    Tw2 = lastKf.Twc_ * initT12;

    const bool success = inliers > static_cast<int>(pts1.size() * 0.75);
    if (!success) {
        cout << fmt::format("match pair num: {}, inliers: {}, ratio: {}.\n",
                            pts1.size(), inliers,
                            static_cast<double>(inliers / pts1.size()));
    }

    return success;
#else
    Pose trueT12 = lastKf.priorTwc_.Inverse() * curF.priorTwc_;

#if DEBUG_POSE_GRAPH_SE3
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<double> distrbution_dist(0., 0.05);
    normal_distribution<double> distrbution_ang(0., 5.0 * kDeg2Rad);
    Pose DisturbT;
    DisturbT.q_wb_ = Eigen::Quaterniond(
        Eigen::AngleAxisd(distrbution_ang(rd), Eigen::Vector3d::UnitX()));
    DisturbT.t_wb_ =
        Eigen::Vector3d(distrbution_dist(rd), 0, distrbution_dist(rd));
    trueT12 = trueT12 * DisturbT;
    Tw2 = lastKf.Twc_ * trueT12;
#else
    trueT12.t_wb_ *= kScaleDriftRatio * (rand() % 5);
    Tw2 = lastKf.Twc_ * trueT12;
    // Tw2 = lastKf.priorTwc_ * trueT12;
#endif

    return true;
#endif
}

bool Sim3PoseGraphOptimization(vector<KeyFrame*>& allKeyframe, int fixedIndex,
                               int loopClosureIndex) {
    Sim3Pose relativeSim3T12;
    if (!CalculateRelativeSim3Transform(allKeyframe[fixedIndex],
                                        allKeyframe[loopClosureIndex],
                                        relativeSim3T12)) {
        cout << "CalculateRelativeSim3Transform failed!" << endl;
        return false;
    } else {
        cout << "fixed and loop closure frame sim3 sT12: "
             << relativeSim3T12.QwbString() << ", "
             << relativeSim3T12.PwbString() << endl;
    }

    // 构建位姿图
    // 1. 选择闭环内的帧
    vector<KeyFrame*> selectKFresult;
    SelectKeyframeInLoopClosure(allKeyframe, fixedIndex, loopClosureIndex,
                                selectKFresult);
    cout << "Select selectKFresult size: " << selectKFresult.size() << endl;

    // 2. 保留帧间位姿先验
    vector<Sim3Pose> loopClosurePoseTwc;
    vector<Sim3Pose> sT12Constraint;
    CalculateLoopClosureSim3PoseAndConstraint(
        relativeSim3T12, selectKFresult, loopClosurePoseTwc, sT12Constraint);
    ofstream of;
    of.open(kBeforeLoopClosurePoseFilePath);
    for (size_t i = 0; i < loopClosurePoseTwc.size(); ++i) {
        cout << "init sTwc[" << i << "]: " << loopClosurePoseTwc[i].QwbString()
             << ", " << loopClosurePoseTwc[i].PwbString() << endl;
        of << loopClosurePoseTwc[i].DebugOutputPoseMessage() << endl;
    }
    of.close();

    for (size_t i = 0; i < sT12Constraint.size(); ++i) {
        cout << fmt::format("constraint[{}]: {}, {}", i,
                            sT12Constraint[i].QwbString(),
                            sT12Constraint[i].PwbString())
             << endl;
    }
    cout << "Construct sT12Constraint size: " << sT12Constraint.size() << endl;

    // 3. 构建信息矩阵和梯度进行迭代优化
    Optimizer::ResidualInfo lastCostInfo =
        CalculatePoseGraphResidualInfo(loopClosurePoseTwc, sT12Constraint);

    Optimizer::ResidualInfo firstCostInfo = lastCostInfo;

    constexpr int kMaxIterativeTime = 100;
    double lambda = 1.0;
    bool acceptNewVariableStatus = true;
    // 包含首帧雅可比为0, H, g维度应该在外部就设置好
    const int optVarNum = loopClosurePoseTwc.size() * 7;
#if USE_SPARSE_H_MATRIX
    Eigen::SparseMatrix<double> H;
    vector<Eigen::Triplet<double>> triplets;  // 三元组用于高效构建稀疏矩阵
    const int num_constraints = loopClosurePoseTwc.size();  // 根据实际情况调整
    triplets.reserve(num_constraints * 4 * 7 * 7);
    H.resize(optVarNum, optVarNum);
#else
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(optVarNum, optVarNum);
#endif
    Eigen::VectorXd g = Eigen::VectorXd::Zero(optVarNum);
    Eigen::VectorXd deltaX(optVarNum);
    chrono::steady_clock::time_point tStart = chrono::steady_clock::now();
    double constructHandGtime = 0, sloveLDLTtime = 0, predictReductionTime = 0;

    for (size_t i = 0; i < kMaxIterativeTime; ++i) {
        chrono::steady_clock::time_point t0 = chrono::steady_clock::now();
        if (acceptNewVariableStatus) {
// 只在状态量更新时需要重新计算信息矩阵和梯度，以节省大量的计算时间
#if USE_SPARSE_H_MATRIX
            ConstructPoseGraphSparseHessianandGradiant(
                loopClosurePoseTwc, sT12Constraint, triplets, H, g);

#else
            ConstructPoseGraphHessianandGradiant(loopClosurePoseTwc,
                                                 sT12Constraint, H, g);
#endif
        }

        // 求解增量
        for (int i = 0; i < H.rows(); ++i) {
#if USE_SPARSE_H_MATRIX
            H.coeffRef(i, i) += lambda;
#else
            H(i, i) += lambda;
#endif
        }
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        constructHandGtime += ChronoMillisecTimeDuration(t0, t1);

#if USE_SPARSE_H_MATRIX
        // 使用适合稀疏矩阵的求解器
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(H);
        if (solver.info() != Eigen::Success) {
            // 分解失败，可以尝试增加lambda或使用其他求解器
            cout << "Error while decompose sparse matrix H!" << endl;
            lambda *= 2.0;
            continue;
        }
        deltaX = solver.solve(g);
#else
        deltaX = H.ldlt().solve(g);
#endif
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        sloveLDLTtime += ChronoMillisecTimeDuration(t1, t2);

        // 保留状态备份并使用增量更新状态
        for (size_t i = 0; i < loopClosurePoseTwc.size(); ++i) {
            loopClosurePoseTwc[i].CopyStatus();
            const auto& delta_i = deltaX.segment<7>(i * 7);
            const auto& delta_q = delta_i.segment<3>(0);
            const auto& delta_p = delta_i.segment<3>(3);
            const auto& delta_s = delta_i(6);
            loopClosurePoseTwc[i].Update(delta_q, delta_p, delta_s);
        }

        // 计算新状态的残差
        Optimizer::ResidualInfo newCostInfo =
            CalculatePoseGraphResidualInfo(loopClosurePoseTwc, sT12Constraint);

        chrono::steady_clock::time_point t3 = chrono::steady_clock::now();

// 计算线性化理论下降值以及真实下降值
#if USE_SPARSE_H_MATRIX
        const double predictCostDecrease =
            ComputePredictionReductionSparseHessian(lambda, deltaX, g, H);
#else
        const double predictCostDecrease =
            ComputePredictionReduction(lambda, deltaX, g, H);
#endif
        chrono::steady_clock::time_point t4 = chrono::steady_clock::now();
        predictReductionTime += ChronoMillisecTimeDuration(t3, t4);

        const double trueCostDecrease = lastCostInfo.cost - newCostInfo.cost;
        cout << fmt::format(
                    "{}th iterative lastCostInfo: {}, newCostInfo: {}, "
                    "predictCostDecrease: "
                    "{}, trueCostDecrease: {}",
                    i, lastCostInfo.cost, newCostInfo.cost, predictCostDecrease,
                    trueCostDecrease)
             << endl;

        // 更新lambda
        const double rho = trueCostDecrease / predictCostDecrease;
        if (rho > 0) {
            if (rho > 0.75) {
                lambda = max(0.3 * lambda, 1e-9);
            } else {
                lambda = max(0.99 * lambda, 1e-9);
            }
            acceptNewVariableStatus = true;
        } else {
            lambda *= 1.5;
            acceptNewVariableStatus = false;
        }

        // 更新状态量
        if (acceptNewVariableStatus) {
            // cout << "H:\n"
            //      << H.diagonal().transpose() << "\n"
            //      << "g: " << g.transpose() << "\n"
            //      << "deltaX: " << deltaX.transpose() << endl;

            cout << fmt::format(
                        "accept iterative {}th time, last cost: {}, new cost: "
                        "{}, "
                        "decrease cost: {}, lambda: {}",
                        i, lastCostInfo.cost, newCostInfo.cost,
                        trueCostDecrease, lambda)
                 << endl;
            lastCostInfo = newCostInfo;
        } else {
            for (auto& sTwc : loopClosurePoseTwc) {
                sTwc.BackUpStatus();
            }
        }
    }
    chrono::steady_clock::time_point tEnd = chrono::steady_clock::now();

    cout << fmt::format(
                "First cost: {:.3f}, last cost: {:.3f}, "
                "decrease cost ratio: {:.3f}, lambda: {:.3f}, total opt spend "
                "time: {:.1f}ms, constructHandGtime: {:.1f}ms, sloveLDLTtime: "
                "{:.1f}ms, "
                "predictReductionTime: {:.1f}ms",
                firstCostInfo.cost, lastCostInfo.cost,
                (firstCostInfo.cost - lastCostInfo.cost) / firstCostInfo.cost,
                lambda, ChronoMillisecTimeDuration(tStart, tEnd),
                constructHandGtime, sloveLDLTtime, predictReductionTime)
         << endl;

    // 4. 传导变换到非回环内帧
    of.open(kClosurePoseFilePath);
    for (size_t i = 0; i < loopClosurePoseTwc.size(); ++i) {
        cout << "sTwc[" << i << "]: " << loopClosurePoseTwc[i].QwbString()
             << ", " << loopClosurePoseTwc[i].PwbString() << endl;
        of << loopClosurePoseTwc[i].DebugOutputPoseMessage() << endl;
    }
    of.close();
    return true;
}

bool CalculateRelativeSim3Transform(const KeyFrame* fixedKF,
                                    const KeyFrame* loopClosureKF,
                                    Sim3Pose& relativeSim3T12) {
    // 设t系为无累积飘移误差的参考系，w系为有累积尺度漂移的参考系，
    // 已知T_tc1和T_wc1，那么尺度漂移scale = P_wc1.norm() / P_tc1.norm()
    // 需要求Sim3变换 sT_wt，使得：
    // T_wc1 = sT_wt * T_tc1，易得
    // sT_wt = T_wc1 * T_tc1.inv(但是这是SE3的式子，不能等价吧？)
    // s = P_wc1.norm() / P_tc1.norm();
    // R_wt = R_wc1 * R_tc1.inv
    // P_wt = 0(因为w系与t系重合)
    // 因此，有：
    //         |s*R_wt, 0|
    // sT_wt = |     0, 1|
    // 那么，在t系下，相机系Tc1c2变换到w系下为：
    //
    // sT_wc2 = sT_wt * Tc1c2

#if DEBUG_POSE_GRAPH_SE3 || 1
    const Pose Tc1c2_t =
        fixedKF->priorTwc_.Inverse() * loopClosureKF->priorTwc_;
    relativeSim3T12 = Sim3Pose(Tc1c2_t, 1.0);
#else

    auto GetSim3PoseTwc = [](const Pose& T_wc, const Pose& T_tc) -> Sim3Pose {
        const double scale = T_wc.t_wb_.norm() / T_tc.t_wb_.norm();
        const Eigen::Quaterniond q_wt = T_wc.q_wb_ * T_tc.q_wb_.inverse();
        const Eigen::Vector3d p_wt = Eigen::Vector3d::Zero();
        Sim3Pose sT_wt(q_wt, p_wt, scale);
        cout << fmt::format("KF id: {}, sT_wt: {}, {}", 1, sT_wt.QwbString(),
                            sT_wt.PwbString())
             << endl;
        return sT_wt * Sim3Pose(T_tc, 1.0);
    };
    const Sim3Pose sTwc1 = GetSim3PoseTwc(fixedKF->Twc_, fixedKF->priorTwc_);
    const Sim3Pose sTwc2 =
        GetSim3PoseTwc(loopClosureKF->Twc_, loopClosureKF->priorTwc_);
    relativeSim3T12 = sTwc1.Inverse() * sTwc2;
#endif

    const double scale = relativeSim3T12.scale_;
    // const Pose Tc1c2_t =
    //     fixedKF->priorTwc_.Inverse() * loopClosureKF->priorTwc_;
    // relativeSim3T12.q_wb_ = q_wt * Tc1c2_t.q_wb_;
    // relativeSim3T12.t_wb_ = scale * Tc1c2_t.t_wb_;
    // relativeSim3T12.scale_ = scale;

    // const Pose Tc1c2_w = fixedKF->Tcw_ * loopClosureKF->Twc_;
    // const Eigen::Vector3d p_t = Tc1c2_t.t_wb_;
    // const Eigen::Vector3d p_w = Tc1c2_w.t_wb_;
    // const double scale = Tc1c2_w.t_wb_.norm() / Tc1c2_t.t_wb_.norm();
    if (scale > 0) {
        //     // |a|*|b|*sin(θ)
        //     const Eigen::Vector3d rot_ang_axis =
        //         p_w.normalized().cross(p_t.normalized());

        //     const Eigen::Matrix3d R_drift =
        //         Eigen::AngleAxisd(asin(rot_ang_axis.norm()),
        //                           rot_ang_axis.normalized())
        //             .toRotationMatrix();
        //     cout << "R_drift:\n"
        //          << R_drift << endl;  // 按道理在R不加噪声时，应该为1

        // relativeSim3T12.q_wb_ = Tc1c2_w.q_wb_ * Eigen::Quaterniond(R_drift);
        // relativeSim3T12.q_wb_ = Tc1c2_w.q_wb_;
        // relativeSim3T12.t_wb_ = Tc1c2_w.t_wb_;
        // relativeSim3T12.scale_ = 1.0;
        const Pose relativeSE3T12 =
            fixedKF->Twc_.Inverse() * loopClosureKF->Twc_;
        cout << "relativeSim3T12: " << relativeSim3T12.QwbString() << ", "
             << relativeSim3T12.PwbString() << "\n"
             << "relativeSE3T12: " << relativeSE3T12.QwbString() << ", "
             << relativeSE3T12.PwbString() << endl;
        return true;
    }
    return false;
}

int CalculateLoopClosureSim3PoseAndConstraint(
    const Sim3Pose& relativeSim3T12, const vector<KeyFrame*>& selectKFresult,
    vector<Sim3Pose>& loopClosurePoseTwc,
    vector<Sim3Pose>& relativePoseConstraint) {

    // 初始化各关键帧的sim3 pose
    loopClosurePoseTwc.reserve(selectKFresult.size());
    for (size_t i = 0; i < selectKFresult.size(); ++i) {
        loopClosurePoseTwc.emplace_back(
            Sim3Pose(selectKFresult[i]->Twc_, 1.0 + 0.1 * i));
        loopClosurePoseTwc.back().debugTimestamp_ =
            selectKFresult[i]->timestamp_;
    }

    // 添加连续帧间相对位姿约束
    relativePoseConstraint.reserve((loopClosurePoseTwc.size()));
    for (size_t i = 1; i < loopClosurePoseTwc.size(); ++i) {
        // const Sim3Pose Twc1 = loopClosurePoseTwc[i - 1];
        // const Sim3Pose Twc2 = loopClosurePoseTwc[i];
        // relativePoseConstraint.emplace_back(Twc1.Inverse() * Twc2);
        relativePoseConstraint.emplace_back(
            selectKFresult[i - 1]->priorTwc_.Inverse() *
                selectKFresult[i]->priorTwc_,
            1.0);
    }
    // 添加回环首、末帧约束，这里添加的是T21作为先验约束
    relativePoseConstraint.emplace_back(relativeSim3T12);
    return relativePoseConstraint.size();
}

double RelativePoseHuberLoss(const Eigen::Matrix<double, 7, 1>& residual) {
    Eigen::Matrix<double, 7, 1> w;
    w << kW[0], kW[0], kW[0], kW[1], kW[1], kW[1], kW[2], kW[2], kW[2];
    Eigen::Matrix<double, 7, 1> residual_multiple_w = w.cwiseProduct(residual);
    return 0.5 * residual_multiple_w.dot(residual);
}

Eigen::Matrix<double, 7, 1> Sim3PoseResidual(const Sim3Pose& sT12,
                                             const Sim3Pose& sTprior12) {
    Eigen::Matrix<double, 7, 1> residual;
    // |s1R1, t1|   |s2R2, t2|
    // |  0,  1 | * |   0,  1| =
    //
    // |s1*s2*R1*R2, s1R1*t2+t1|
    // |          0, 1         |
    // ΔR = LogSO3(R1 * R2)
    // Δt = s1*R1*t2 + t1
    // 需保证和雅可比计算的顺序一致
    residual.segment<3>(0) =
        LogSO3((sTprior12.q_wb_.inverse() * sT12.q_wb_).toRotationMatrix());

    residual.segment<3>(3) = sT12.t_wb_ - sTprior12.t_wb_;

    // 注意：这里实现存在的问题是量纲不统一，且scale一定是非负数
    residual[6] =
        sT12.scale_ - sTprior12.scale_;  // 相邻帧间尺度漂移比例应该接近于1.0

#if DEBUG_POSE_GRAPH_SE3
    residual[6] = 0.;
#endif

    return residual;
}

double ComputePredictionReduction(const double lambda,
                                  const Eigen::VectorXd& deltaX,
                                  const Eigen::VectorXd& g,
                                  const Eigen::MatrixXd& H) {
    // 实际下降值为： lastCost - newCost
    // g = -J.T * r
    return -0.5 * deltaX.dot(H * deltaX) + deltaX.dot(g) -
           0.5 * lambda * deltaX.squaredNorm();
}

double ComputePredictionReductionSparseHessian(
    const double lambda, const Eigen::VectorXd& deltaX,
    const Eigen::VectorXd& g, const Eigen::SparseMatrix<double>& H) {
    // 实际下降值为： lastCost - newCost
    // g = -J.T * r
    return -0.5 * deltaX.dot(H * deltaX) + deltaX.dot(g) -
           0.5 * lambda * deltaX.squaredNorm();
}

Optimizer::ResidualInfo CalculatePoseGraphResidualInfo(
    const vector<Sim3Pose>& sTwc, const vector<Sim3Pose>& sT12Constraint) {
    Optimizer::ResidualInfo info;
    for (size_t i = 1; i < sTwc.size(); ++i) {
        const Sim3Pose sTwc1 = sTwc[i - 1];
        const Sim3Pose sTwc2 = sTwc[i];
        // 连续帧间
        const Sim3Pose sT12 = sTwc1.Inverse() * sTwc2;
        // 注意检查下标索引是否正确
        const Eigen::Matrix<double, 7, 1> res =
            Sim3PoseResidual(sT12, sT12Constraint[i - 1]);
        info.cost += RelativePoseHuberLoss(res);  // 使用huber loss替代
        ++info.totalConstraintNum;
    }
    // 闭环帧间
    const Sim3Pose sT12 = sTwc.front().Inverse() * sTwc.back();
    const Eigen::Matrix<double, 7, 1> res =
        Sim3PoseResidual(sT12, sT12Constraint.back());
    info.cost += RelativePoseHuberLoss(res);
    ++info.totalConstraintNum;
    return info;
}

void ConstructPoseGraphHessianandGradiant(
    const vector<Sim3Pose>& sTwc, const vector<Sim3Pose>& sT12Constraint,
    Eigen::MatrixXd& H, Eigen::VectorXd& g) {
    // 只选择在回环内的KF进行位姿图优化，回环外的仅根据连接关系进行更新
    constexpr int Sim3Dim = 7;
    H.setZero();
    g.setZero();
    const double w[3] = {sqrt(kW[0]), sqrt(kW[1]), sqrt(kW[2])};

    // 构建信息矩阵H与梯度g
    Eigen::Matrix<double, 7, 7> A1, A2;
    auto FillHessianAndGradiant = [&H, &g, &w, &sTwc, &sT12Constraint, &A1,
                                   &A2](const int i1, const int i2,
                                        const int constraintId) -> void {
        const Eigen::Matrix3d Rw1 = sTwc[i1].q_wb_.toRotationMatrix();
        const Eigen::Vector3d& Pw1 = sTwc[i1].t_wb_;
        const double s1 = sTwc[i1].scale_;

        const Eigen::Matrix3d Rw2 = sTwc[i2].q_wb_.toRotationMatrix();
        const Eigen::Vector3d& Pw2 = sTwc[i2].t_wb_;
        const double s2 = sTwc[i2].scale_;

        // 注意取逆
        const Eigen::Matrix3d dR =
            sT12Constraint[constraintId].q_wb_.toRotationMatrix().transpose();
        const Eigen::Vector3d& dP = sT12Constraint[constraintId].t_wb_;
        const double ds = sT12Constraint[constraintId].scale_;

        // ΔR = LogSO3(dR * Rw1.inv * Rw2)
        // ΔT = Twc1.inv * Twc2
        // ΔP = 1/s1 * Rw1.inv * Pw2 - 1/s1 * Rw1.inv * Pw1 - dP
        //    = 1/s1 * Rw1.inv * (Pw2 - Pw1) - dP
        // Δs = s1*s2 - ds
        // 计算残差向量
        Eigen::Matrix<double, 7, 1> residual;
        residual.segment<3>(0) = LogSO3(dR * Rw1.transpose() * Rw2);
        const double invS1 = 1.0 / s1;
        const Eigen::Vector3d dPw = Pw2 - Pw1;
        residual.segment<3>(3) = invS1 * Rw1.transpose() * dPw - dP;
        residual(6) = s1 * s2 - ds;

#if DEBUG_POSE_GRAPH_SE3
        residual(6) = 0.;
#endif

        A1.setZero();
        A2.setZero();
        const Eigen::Matrix3d invJr =
            InverseRightJacobianSO3(residual.segment<3>(0));

        // 使用"BCH近似"之前，需要通过"伴随性质"将扰动量换到右边
        // dLogSO3(dR * R1.T*R2) ---> 微分扰动
        // = LogSO3(dR * exp(-ε1^)*R1.T*R2) ---> 使用伴随: Exp(ε1)*R = R * Exp(R.T * ε1)
        // = LogSO3(dR*R1.T*R2 * Exp(R2.T*R1*-ε1)) ---> Exp{小量}，使用BCH近似
        // = [Jr(dR*R1.T*R2).inv * -R2.T*R1*ε1] + LogSo3(dR*R1.T*R2)
        // 关于Rw2，易得最终的BCH近似为：
        // [Jr(dR*R1.T*R2).inv * ε2] + LogSo3(dR*R1.T*R2)

        // ΔR w.r.t Rw1
        A1.block<3, 3>(0, 0) = -invJr * Rw2.transpose() * Rw1 * w[0];
        // ΔR w.r.t Rw2
        A2.block<3, 3>(0, 0) = invJr * dR.transpose() * w[0];
        // ΔR w.r.t Pw1, s1, Pw2, s2 = 0

        // ΔP w.r.t Rw1
        A1.block<3, 3>(3, 0) =
            invS1 * SkewSymmetric(Rw1.transpose() * dPw) * w[1];
        // ΔP w.r.t Rw2 = 0
        // ΔP w.r.t Pw1
        A1.block<3, 3>(3, 3) = -invS1 * Rw1.transpose() * w[1];
        // ΔP w.r.t Pw2
        A2.block<3, 3>(3, 3) = invS1 * Rw1.transpose() * w[1];
        // ΔP w.r.t s1
        A1.block<3, 1>(3, 6) = -invS1 * invS1 * Rw1.transpose() * dPw * w[1];
        // ΔP w.r.t s2 = 0

        // Δs w.r.t Rw1, Rw2, Pw1, Pw2 = 0
        // Δs w.r.t s1
        A1(6, 6) = s2 * w[2];
        // Δs w.r.t s2
        A2(6, 6) = s1 * w[2];

#if DEBUG_POSE_GRAPH_SE3
        A1.block<3, 1>(3, 6).setZero();
        A1(6, 6) = 0;
        A2(6, 6) = 0;
#endif

        if (i1 == 0) {
            A1.setZero();
        }

        // A1及A2在雅可比矩阵中的起始位置aj1, aj2
        //     sTw1  sTw2, sTw3... sTwn
        // J =
        //
        const int aj1 = i1 * Sim3Dim;
        const int aj2 = i2 * Sim3Dim;
        // 信息矩阵叠加雅可比J.T*J信息，注意，J是稀疏的，因此只需叠加当前的A1, A2而不必使用整个雅可比J计算
        // 若使用J，则只需加1次，即 H+=J.T * J，但这里使用J的分块将有4次填充
        H.block<7, 7>(aj1, aj1) += A1.transpose() * A1;
        H.block<7, 7>(aj1, aj2) += A1.transpose() * A2;
        H.block<7, 7>(aj2, aj2) += A2.transpose() * A2;
        H.block<7, 7>(aj2, aj1) += A2.transpose() * A1;

        g.segment<7>(aj1) -= A1.transpose() * residual;
        g.segment<7>(aj2) -= A2.transpose() * residual;
    };

    for (size_t i = 1; i < sTwc.size(); ++i) {
        FillHessianAndGradiant(i - 1, i, i - 1);
    }

    // 闭环残差填充
    FillHessianAndGradiant(0, sTwc.size() - 1, sTwc.size() - 1);
}

void ConstructPoseGraphSparseHessianandGradiant(
    const vector<Sim3Pose>& sTwc, const vector<Sim3Pose>& sT12Constraint,
    vector<Eigen::Triplet<double>>& triplets, Eigen::SparseMatrix<double>& H,
    Eigen::VectorXd& g) {
    // 只选择在回环内的KF进行位姿图优化，回环外的仅根据连接关系进行更新
    constexpr int Sim3Dim = 7;
    triplets.clear();
    g.setZero();
    const double w[3] = {sqrt(kW[0]), sqrt(kW[1]), sqrt(kW[2])};

    // 构建信息矩阵H与梯度g
    Eigen::Matrix<double, 7, 7> A1, A2;
    auto FillHessianAndGradiant = [&triplets, &g, &w, &sTwc, &sT12Constraint,
                                   &A1, &A2](const int i1, const int i2,
                                             const int constraintId) -> void {
        const Eigen::Matrix3d Rw1 = sTwc[i1].q_wb_.toRotationMatrix();
        const Eigen::Vector3d& Pw1 = sTwc[i1].t_wb_;
        const double s1 = sTwc[i1].scale_;

        const Eigen::Matrix3d Rw2 = sTwc[i2].q_wb_.toRotationMatrix();
        const Eigen::Vector3d& Pw2 = sTwc[i2].t_wb_;
        const double s2 = sTwc[i2].scale_;

        // 注意取逆
        const Eigen::Matrix3d dR =
            sT12Constraint[constraintId].q_wb_.toRotationMatrix().transpose();
        const Eigen::Vector3d& dP = sT12Constraint[constraintId].t_wb_;
        const double ds = sT12Constraint[constraintId].scale_;

        // ΔR = LogSO3(dR * Rw1.inv * Rw2)
        // ΔT = Twc1.inv * Twc2
        // ΔP = 1/s1 * Rw1.inv * Pw2 - 1/s1 * Rw1.inv * Pw1 - dP
        //    = 1/s1 * Rw1.inv * (Pw2 - Pw1) - dP
        // Δs = s1*s2 - ds
        // 计算残差向量
        Eigen::Matrix<double, 7, 1> residual;
        residual.segment<3>(0) = LogSO3(dR * Rw1.transpose() * Rw2);
        const double invS1 = 1.0 / s1;
        const Eigen::Vector3d dPw = Pw2 - Pw1;
        residual.segment<3>(3) = invS1 * Rw1.transpose() * dPw - dP;
        residual(6) = s1 * s2 - ds;

#if DEBUG_POSE_GRAPH_SE3
        residual(6) = 0.;
#endif

        A1.setZero();
        A2.setZero();
        const Eigen::Matrix3d invJr =
            InverseRightJacobianSO3(residual.segment<3>(0));

        // 使用"BCH近似"之前，需要通过"伴随性质"将扰动量换到右边
        // dLogSO3(dR * R1.T*R2) ---> 微分扰动
        // = LogSO3(dR * exp(-ε1^)*R1.T*R2) ---> 使用伴随: Exp(ε1)*R = R * Exp(R.T * ε1)
        // = LogSO3(dR*R1.T*R2 * Exp(R2.T*R1*-ε1)) ---> Exp{小量}，使用BCH近似
        // = [Jr(dR*R1.T*R2).inv * -R2.T*R1*ε1] + LogSo3(dR*R1.T*R2)
        // 关于Rw2，易得最终的BCH近似为：
        // [Jr(dR*R1.T*R2).inv * ε2] + LogSo3(dR*R1.T*R2)

        // ΔR w.r.t Rw1
        A1.block<3, 3>(0, 0) = -invJr * Rw2.transpose() * Rw1 * w[0];
        // ΔR w.r.t Rw2
        A2.block<3, 3>(0, 0) = invJr * dR.transpose() * w[0];
        // ΔR w.r.t Pw1, s1, Pw2, s2 = 0

        // ΔP w.r.t Rw1
        A1.block<3, 3>(3, 0) =
            invS1 * SkewSymmetric(Rw1.transpose() * dPw) * w[1];
        // ΔP w.r.t Rw2 = 0
        // ΔP w.r.t Pw1
        A1.block<3, 3>(3, 3) = -invS1 * Rw1.transpose() * w[1];
        // ΔP w.r.t Pw2
        A2.block<3, 3>(3, 3) = invS1 * Rw1.transpose() * w[1];
        // ΔP w.r.t s1
        A1.block<3, 1>(3, 6) = -invS1 * invS1 * Rw1.transpose() * dPw * w[1];
        // ΔP w.r.t s2 = 0

        // Δs w.r.t Rw1, Rw2, Pw1, Pw2 = 0
        // Δs w.r.t s1
        A1(6, 6) = s2 * w[2];
        // Δs w.r.t s2
        A2(6, 6) = s1 * w[2];

#if DEBUG_POSE_GRAPH_SE3
        A1.block<3, 1>(3, 6).setZero();
        A1(6, 6) = 0;
        A2(6, 6) = 0;
#endif

        if (i1 == 0) {
            A1.setZero();
        }

        // A1及A2在雅可比矩阵中的起始位置aj1, aj2
        //     sTw1  sTw2, sTw3... sTwn
        // J =
        //
        const int aj1 = i1 * Sim3Dim;
        const int aj2 = i2 * Sim3Dim;
        // 信息矩阵叠加雅可比J.T*J信息，注意，J是稀疏的，因此只需叠加当前的A1, A2而不必使用整个雅可比J计算
        // 若使用J，则只需加1次，即 H+=J.T * J，但这里使用J的分块将有4次填充
        // H.block<7, 7>(aj1, aj1) += A1.transpose() * A1;
        EmplaceBackTriplet<7, 7>(aj1, aj1, A1.transpose() * A1, triplets);

        // H.block<7, 7>(aj1, aj2) += A1.transpose() * A2;
        EmplaceBackTriplet<7, 7>(aj1, aj2, A1.transpose() * A2, triplets);

        // H.block<7, 7>(aj2, aj2) += A2.transpose() * A2;
        EmplaceBackTriplet<7, 7>(aj2, aj2, A2.transpose() * A2, triplets);

        // H.block<7, 7>(aj2, aj1) += A2.transpose() * A1;
        EmplaceBackTriplet<7, 7>(aj2, aj1, A2.transpose() * A1, triplets);

        g.segment<7>(aj1) -= A1.transpose() * residual;
        g.segment<7>(aj2) -= A2.transpose() * residual;
    };

    for (size_t i = 1; i < sTwc.size(); ++i) {
        FillHessianAndGradiant(i - 1, i, i - 1);
    }

    // 闭环残差填充
    FillHessianAndGradiant(0, sTwc.size() - 1, sTwc.size() - 1);

    // 从三元组构建稀疏矩阵
    H.setFromTriplets(triplets.begin(), triplets.end());
    H.makeCompressed();  // 压缩存储格式
    for (size_t i = 0; i < triplets.size(); ++i) {
        if (abs(triplets[i].value()) < 1e-12) {
            cout << "triplets[" << i << "]: " << triplets[i].value() << endl;
        }
    }
    // cout << "Sparse matrix H:\n" << H << endl;
}

bool SelectKeyframeInLoopClosure(vector<KeyFrame*>& allKeyframe, int fixedIndex,
                                 int loopClosureIndex,
                                 vector<KeyFrame*>& selectResult) {
    const int fixKFid = allKeyframe[fixedIndex]->id_;
    const int loopClosureKFid = allKeyframe[loopClosureIndex]->id_;

    selectResult.reserve(allKeyframe.size());
    for (KeyFrame* kf : allKeyframe) {
        if (kf->id_ < fixKFid || kf->id_ > loopClosureKFid) {
            continue;
        }
        selectResult.emplace_back(kf);
    }

    // fixed帧为首帧
    sort(selectResult.begin(), selectResult.end(),
         [](const KeyFrame* f1, const KeyFrame* f2) {
             return f1->id_ < f2->id_;
         });

    if (selectResult.front()->id_ != fixKFid ||
        selectResult.back()->id_ != loopClosureKFid) {
        cout << fmt::format(
            "Error while collect loop closure, except kf id range: [{}, {}], "
            "result range: [{}, {}]\n",
            fixKFid, loopClosureKFid, selectResult.front()->id_,
            selectResult.back()->id_);
        return false;
    }

    return true;
}

bool UmeyamaSim3Transform(const vector<Eigen::Vector3d>& ps1,
                          const vector<Eigen::Vector3d>& ps2, Sim3Pose& T12) {

    // 1. lightglue匹配得到一对匹配点集
    // 2. 对应匹配点集的3D坐标已知，但是存在尺度漂移，
    // 各获取一组3D点，ps1, ps2，其中，点集应该满足：
    // p1 = s*R*p2 + t12，由此获取Sim3变换初值
    // 3. 使用非线性优化精炼闭环的Sim3相对变换

    return false;
}
