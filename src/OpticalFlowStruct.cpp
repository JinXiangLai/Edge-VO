#include "OpticalFlowStruct.h"

using namespace std;
void OpticalFlowStruct::RemoveUselessLandmark() {
    std::vector<cv::Point2f> tempPts;
    std::vector<Landmark*> tempLandmark;
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
        "optflw remove useless landmark num: {}, remain landmark num: {}\n",
        totalFeatureCreated_ - tempPts.size(), tempPts.size());
    // totalFeatureCreated_ = tempPts.size(); // 不能重新赋值总的特征，因为跟踪过程会丢失

    prevPts_ = std::move(tempPts);
    trackLandmark_ = std::move(tempLandmark);
}