#ifndef CLASS_INITIALIZER
#define CLASS_INITIALIZER

#include <Optimizer.h>

class Initializer {
   public:
    Initializer(const std::shared_ptr<Camera>& cam, Optimizer* optimizer);
    ~Initializer();
    bool InitializeSecondKeyFramePose(const int findMatchNum,
                                      const double findMatchRatio,
                                      KeyFrame& curF);

    bool ConstructAndDecomposeEssentialMatrix(
        std::vector<Eigen::Vector4d>& uv2obv, Pose& result);

    bool ConstructAndDecomposeEssentialMatrixOpenCV(
        std::vector<Eigen::Vector4d>& uv2obv, Pose& result);

    bool CheckInitPose(const int selectNum, const Pose& T12,
                       std::vector<Eigen::Vector4d>& uv2obv);

    void WriteDebugInitImage();

    void GenerateDebugImage(const cv::Mat& curImg);

    void DrawMatchPoint(const Eigen::Vector2d& kp1, const Eigen::Vector2d& kp2);

    void PutText2DebugMatchImg(const std::string& info, const int writeRow);

    cv::VideoWriter debugInitTrackWriter_;
    cv::Mat debugMatchImg_;

   private:
    std::shared_ptr<Camera> cam_;
    Optimizer* optimizer_;
};
#endif