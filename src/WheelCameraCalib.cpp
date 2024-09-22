#include "WheelCameraCalib.h"

using namespace std;

WheelCameraCalib::WheelCameraCalib() {
    q_cg_.normalize();
    Pose Tcg(q_cg_, t_cg_);
    Pose Tgc = Tcg.Inverse();
    Eigen::Vector3d t_vc = Tgc.t_wb_;
    t_vc.z() -= radius_;
    Tvc_ = Pose(Tgc.q_wb_, t_vc);
    Tcv_ = Tvc_.Inverse();
}
