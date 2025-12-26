#include "OpticalFlowStruct.h"

#include "Utils.h"
using namespace std;

// 全局变量定义
std::mutex globalOptFlwMutex;
OpticalFlowStruct globalOptFlw;
shared_ptr<SuperPoint> superpointPtr;
shared_ptr<LightGlue> lightgluePtr;

void OpticalFlowStruct::RemoveUselessLandmark() {
    std::vector<cv::Point2f> tempPts;
    std::vector<shared_ptr<Landmark>> tempLandmark;
    tempPts.reserve(prevPts_.size());
    tempLandmark.reserve(trackLandmark_.size());

    auto SelectUsefulLandmark = [&tempPts, &tempLandmark, this](
                                    const size_t startIdx,
                                    const size_t endIdx) {
        for (size_t i = startIdx; i < endIdx; ++i) {
            if (trackLandmark_[i] == nullptr ||
                trackLandmark_[i]->CanBeDelete()) {
                continue;
            }
            tempPts.emplace_back(prevPts_[i]);
            tempLandmark.emplace_back(trackLandmark_[i]);
        }
    };

    SelectUsefulLandmark(0, historyLandmarkNum_);
    const size_t srcHistoryNum = historyLandmarkNum_;
    historyLandmarkNum_ = tempPts.size();

    SelectUsefulLandmark(srcHistoryNum, prevPts_.size());
    cout << fmt::format(
        "optical flow remove useless landmark num: {}, remain landmark num: "
        "{}\n",
        totalFeatureCreated_ - tempPts.size(), tempPts.size());
    // totalFeatureCreated_ = tempPts.size(); // 不能重新赋值总的特征，因为跟踪过程会丢失

    prevPts_ = std::move(tempPts);
    trackLandmark_ = std::move(tempLandmark);
}

void OpticalFlowStruct::GetFundamentalMatrixF12(const shared_ptr<Camera> cam,
                                                Eigen::Matrix3f& e) {
    const int selectNum = min(int(prevPts_.size() * 0.9), 500);
    vector<cv::Point2f> ps1, ps2;
    ps1.reserve(selectNum);
    ps2.reserve(selectNum);
    const int sampleStep = prevPts_.size() / selectNum;
    for (int i = 0; i < selectNum; i += sampleStep) {
        if (trackLandmark_[i] == nullptr || trackLandmark_[i]->CanBeDelete()) {
            i -= (sampleStep - 1);
            continue;
        }
        ps1.emplace_back(trackLandmark_[i]->GetLastFrameObvCV());
        ps2.emplace_back(prevPts_[i].x, prevPts_[i].y);
    }

    cv::Mat cvE =
        cv::findEssentialMat(ps2, ps1, Eigen2CVmat(cam->K_[0]), cv::RANSAC);
    cout << "essential E12:\n" << cvE << endl;
    e = (cam->Kinv_[0].transpose() * CVmat2Eigen(cvE) * cam->Kinv_[0])
            .cast<float>();
}
