#ifndef CLASS_OPTIMIZER
#define CLASS_OPTIMIZER

#include <vector>

#include "Config.h"
#include "Landmark.h"
#include "Pose.h"
#include "KeyFrame.h"

class Optimizer {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Optimizer(const std::vector<cv::Mat> &dist, const std::vector<cv::Mat> &dx, const std::vector<cv::Mat> &dy, 
        Camera *cam, const double lambda = 1.0, const int maxIte = 100, const bool useInvDepth = false, 
        const bool onlyPoseUpdate = false);
    
    bool Optimize(std::vector<Landmark*> &_pc1, std::vector<Pose> &T12);

    Eigen::VectorXd CalculateResidual(const std::vector<Landmark*> &pc1, 
        const std::vector<Pose> &T12);

    Eigen::MatrixXd CalculateJacobian(const std::vector<Landmark* > &pc1, 
        const std::vector<Pose> &T12, Eigen::MatrixXd &H, Eigen::VectorXd &b, Eigen::VectorXd &g);

    Eigen::VectorXd SchurCompleteSolve(const Eigen::MatrixXd &H, const Eigen::VectorXd &b, const int poseNum, const int pointNum, 
        const int poseDim = 6, const int pointDim = 1);

    // TODO： 先不考虑边缘化，而是直接丢弃首帧
    void MarginalizeFirstKeyFrame();

    void AddOneKeyFeame(KeyFrame *kf);

    void ResetOptVariables();

    bool ConstructJ_H_b_g();

private:
    double lambda_ = 1.0;
    std::vector<cv::Mat> dist_, dx_, dy_;
    int maxIte_ = 100;
    bool useInvDepth_ = false;
    const bool onlyPoseUpdate_ = false; 
    Camera *cam_;

    std::vector<KeyFrame*> window_;
    Pose *margTwc_ = nullptr;
    std::set<Landmark*> ps_;
    std::vector<Landmark*> optLandmark_; // 投影到最新帧能被观测到的才加入，以减小问题规模
    KeyFrame *oldest_ = nullptr;
    KeyFrame *newest_ = nullptr;
    Eigen::MatrixXd J_, H_;
    Eigen::VectorXd b_, g_;
};

#endif