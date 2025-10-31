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

    bool FindEssentialMatrixRansac(const std::vector<Eigen::Vector4d>& uv2obv,
                                   Eigen::Matrix3d& matrixE,
                                   const double inlinerRatio = 0.75,
                                   const double successProb = 0.95);

    Eigen::MatrixXd ConstructCoffeeMatrix(
        const std::vector<Eigen::Vector2d>& ps1,
        const std::vector<Eigen::Vector2d>& ps2, const Eigen::Matrix3d& normT1,
        const Eigen::Matrix3d& normT2);

    Eigen::Matrix3d GetEssentialMatrix(const Eigen::MatrixXd& coffeMatrix,
                                       const Eigen::Matrix3d& normT1,
                                       const Eigen::Matrix3d& normT2);

    double ComputeEpipolarConstraintRmse(
        const Eigen::Matrix3d& E, const std::vector<Eigen::Vector4d>& uv2obv);

    bool FindRotationAndTranslation(const Eigen::Matrix3d& matrixE,
                                    const std::vector<Eigen::Vector4d>& uv2obv,
                                    Eigen::Matrix3d& R, Eigen::Vector3d& t);

    bool ConstructAndDecomposeEssentialMatrixOpenCV(
        std::vector<Eigen::Vector4d>& uv2obv, Pose& result);

    bool CheckInitPose(const int selectNum, const Pose& T12,
                       std::vector<Eigen::Vector4d>& uv2obv);

    void WriteDebugInitImage();

    void GenerateDebugImage(const cv::Mat& curImg);

    void DrawMatchPoint(const Eigen::Vector2d& kp1, const Eigen::Vector2d& kp2);

    void PutText2DebugMatchImg(const std::string& info, const int writeRow);

    // 计算 Hartley 归一化变换 T，使点集中心在原点、平均距离为 sqrt(2)
    Eigen::Matrix3d ComputeNormalizationTransform(
        const std::vector<Eigen::Vector2d>& pts);

    std::vector<Eigen::Vector2d> ApplyTransform(
        const std::vector<Eigen::Vector2d>& pts, const Eigen::Matrix3d& T);

    Eigen::MatrixXd BuildDesignMatrix(const std::vector<Eigen::Vector2d>& pts1,
                                      const std::vector<Eigen::Vector2d>& pts2);

    Eigen::Matrix3d Vec9ToMat3(const Eigen::VectorXd& v);

    Eigen::Matrix3d EnforceEssentialConstraint(
        const Eigen::Matrix3d& E_initial);

    double ComputeEpipolarRMS(const Eigen::Matrix3d& E,
                              const std::vector<Eigen::Vector2d>& pts1,
                              const std::vector<Eigen::Vector2d>& pts2);

    bool ConstructAndDecomposeEssentialMatrixNormPoint(
        std::vector<Eigen::Vector4d>& uv2obv, Pose& result);

    cv::VideoWriter debugInitTrackWriter_;
    cv::Mat debugMatchImg_;

   private:
    std::shared_ptr<Camera> cam_;
    Optimizer* optimizer_;
};
#endif