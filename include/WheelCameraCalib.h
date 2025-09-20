#ifndef CLASS_WHEEL_CAMERA_CALIB
#define CLASS_WHEEL_CAMERA_CALIB

#include "Config.h"
#include "Pose.h"

class WheelCameraCalib {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    WheelCameraCalib(const std::vector<double>& Qcg,
                     const std::vector<double>& Pcg, const double radius);
    Pose Tcv_, Tvc_;
};

#endif
