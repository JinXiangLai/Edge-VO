#ifndef CLASS_WHEEL_CAMERA_CALIB
#define CLASS_WHEEL_CAMERA_CALIB

#include "Config.h"
#include "Pose.h"

class WheelCameraCalib { 
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    WheelCameraCalib();
    Pose Tcv_, Tvc_;
private:
    // 手动填写
#if CAR_NUM == 15
    Eigen::Quaterniond q_cg_{-0.4975894231104555, -0.505161241707068, 0.5085381266442339, -0.4884729876203498};
    Eigen::Vector3d t_cg_{0.1016387795887079, 1.725914890100064, -1.766470201637028};
#elif CAR_NUM == 18
    Eigen::Quaterniond q_cg_{0.4942495072630623, 0.505820516001542, -0.4988584606367882, 0.5010022618843634};
    Eigen::Vector3d t_cg_{0.06400512447560017, 1.695829476938501, -1.788386848651872};
#endif
    double radius_ = 0.376;
};

#endif
