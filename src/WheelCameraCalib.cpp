#include "WheelCameraCalib.h"

using namespace std;

WheelCameraCalib::WheelCameraCalib(const vector<double>& Qcg,
                                   const vector<double>& Pcg,
                                   const double radius) {
    Eigen::Quaterniond q_cg(Qcg[3], Qcg[0], Qcg[1], Qcg[2]);
    q_cg.normalize();
    Eigen::Vector3d t_cg_(Pcg[0], Pcg[1], Pcg[2]);
    Pose Tcg(q_cg, t_cg_);
    Pose Tgc = Tcg.Inverse();
    Eigen::Vector3d t_vc = Tgc.t_wb_;
    t_vc.z() -= radius;
    Tvc_ = Pose(Tgc.q_wb_, t_vc);
    Tcv_ = Tvc_.Inverse();
}
