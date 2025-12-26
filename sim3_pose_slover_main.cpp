#include <fmt/core.h>
#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <set>

#include "Utils.h"

using namespace std;

struct UnCamera {
    double fx = 520;
    double cx = 320;
    double fy = 520;
    double cy = 255;
    int width = 640;
    int height = 480;
};

typedef Eigen::Matrix<double, 3, Eigen::Dynamic> DynamicPointMatrix;

DynamicPointMatrix GenerateMapPoints(
    const UnCamera& cam, const Eigen::Matrix<double, 2, Eigen::Dynamic>& Px);

void AddNoise2Pw(const double addNoiseNumRatio, DynamicPointMatrix& Pw);

Sim3Pose GenerateSim3Pose(const double r, const double p, const double y,
                          const Eigen::Vector3d& Pwc, const double s);

DynamicPointMatrix TransformPw2Pc(const Sim3Pose& sTwc,
                                  const DynamicPointMatrix& Pw);

int main() {
    UnCamera cam;
    const int selectPointNumEachRow = 10;
    const int selectPointNumEachCol = 5;
    const int colStep = cam.width / selectPointNumEachRow;
    const int rowStep = cam.height / selectPointNumEachCol;
    Eigen::MatrixXd Px(2, selectPointNumEachRow * selectPointNumEachCol);
    Px.setZero();
    int count = 0;
    for (int x = colStep - 10; x < cam.width; x += colStep) {
        for (int y = rowStep - 5; y < cam.height; y += rowStep) {
            Px.col(count++) << x, y;
        }
    }
    // cout << "Px:\n" << Px << endl;

    const Eigen::MatrixXd Pw = GenerateMapPoints(cam, Px);
    // cout << "Pw:\n" << Pw << endl;

    const Sim3Pose sTwc1 = GenerateSim3Pose(5, 2, 3, {0.5, 0.1, 0.2}, 1.0);
    // 从w系转到这里的点会除以scale 2.0
    const Sim3Pose sTwc2 = GenerateSim3Pose(5, 2, 3, {1.0, 0.2, 0.4}, 2.0);
    DynamicPointMatrix Pc1 = TransformPw2Pc(sTwc1, Pw);
    DynamicPointMatrix Pc2 = TransformPw2Pc(sTwc2, Pw);
    // 点集前n个添加噪声
    const double innerRatio = 0.6;
    AddNoise2Pw(1.0 - innerRatio, Pc1);

    Sim3Pose sT12;
    const bool success =
        CalculateSim3PosesT12RANSAC(Pc1, Pc2, sT12, 3, 0.999, innerRatio);
    if (success) {
        cout << "Solve sim3Pose sT12 succeed! sT12.inv * sT12True:\n"
             << sT12.Inverse() * (sTwc1.Inverse() * sTwc2) << "\n"
             << "sT12:\n"
             << sT12 << endl;
    } else {
        cout << "Solve sim3Pose sT12 failed!" << endl;
    }

    return 0;
}

DynamicPointMatrix GenerateMapPoints(
    const UnCamera& cam, const Eigen::Matrix<double, 2, Eigen::Dynamic>& Px) {
    Eigen::MatrixXd Pn(3, Px.cols());
    for (int j = 0; j < Px.cols(); ++j) {
        const auto& px = Px.col(j);
        const double xn = (px[0] - cam.cx) / cam.fx;
        const double yn = (px[1] - cam.cy) / cam.fy;
        Pn.col(j) << xn, yn, 1.0;
    }

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> distDepth(3.0, 10.0);
    for (int j = 0; j < Pn.cols(); ++j) {
        Pn.col(j) *= distDepth(rd);
    }

    return Pn;
}

Sim3Pose GenerateSim3Pose(const double r, const double p, const double y,
                          const Eigen::Vector3d& Pwc, const double s) {
    const Eigen::Vector3d uX(1, 0, 0);
    const Eigen::Vector3d uY(0, 1, 0);
    const Eigen::Vector3d uZ(0, 0, 1);
    const Eigen::Matrix3d Rwc =
        Eigen::AngleAxisd(y * kDeg2Rad, uZ) *
        Eigen::AngleAxisd(p * kDeg2Rad, uY) *
        Eigen::AngleAxisd(r * kDeg2Rad, uX).toRotationMatrix();
    return Sim3Pose(Eigen::Quaterniond(Rwc), Pwc, s);
}

DynamicPointMatrix TransformPw2Pc(const Sim3Pose& sTwc,
                                  const DynamicPointMatrix& Pw) {
    DynamicPointMatrix Pc = Pw;
    const Sim3Pose sTcw = sTwc.Inverse();
    for (int col = 0; col < Pw.cols(); ++col) {
        Pc.col(col) = sTcw * Pw.col(col);
    }
    return Pc;
}

void AddNoise2Pw(const double addNoiseNumRatio, DynamicPointMatrix& Pw) {
    const int noiseNum = int(Pw.cols() * addNoiseNumRatio);
    mt19937 rd;
    normal_distribution<double> dist(0, 5.0);
    for (int j = 0; j < noiseNum; ++j) {
        const Eigen::Vector3d noise(dist(rd), dist(rd), dist(rd));
        Pw.col(j) += noise;
    }
}
