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
    void MarginalizeFirstKeyFrame() {
        /*********************************************************
        * 注意：VINS-MONO论文中的r_p, Hp分别代表先验残差、先验雅可比，
        * 即 先验约束项 |r_p - Hp * X|^2 <==> |r_p - Jp * X|^2
        * 注意：VINS-MONO论文中，多处出现H矩阵，其均不代表J’*J!!!
        * 参考为：https://github.com/StevenCui/VIO-Doc/tree/master
        ***********************************************************
        * |A B|   |dx1|   |g1|
        * |C D| * |dx2| = |g2| ==>
        * 将A边缘化掉，得：
        * |E F|   |dx1|   |h1|
        * |0 G| * |dx2| = |h2| ==>
        * G*dx2 = h2, 并且，dx2满足：
        * {我们知道，对于一个线性化的量测方程而言，有：
        * J'*J * dx = -J' * b，
        ******************************************************
        * 令G = J' * J, h2 = -J' * b, 
        * 那么构建先验约束： r = |b - J*X|^2，该式在求解极小值点dx的过程中，
        * 恰好能满足出现： G*dx = h2这一先验约束
        *********************************************************
        * 所以，只被margTwc观测到的地图点，我们不再用它构建方程，直接丢弃
        * 既被margTwc又被其他Twc观测到的地图点，不将其边缘化，而是继续更新
        *******************************************************/
    }

    void AddOneKeyFeame(KeyFrame *kf) {
        if(window_.size() == kMaxKFnumInWindow) {
            margTwc_ = &window_.front()->Twc_;
            MarginalizeFirstKeyFrame();
        }
    }

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
};

#endif