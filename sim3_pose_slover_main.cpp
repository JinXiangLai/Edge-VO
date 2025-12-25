#include <fmt/core.h>
#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <set>

#include "Pose.h"

using namespace std;

constexpr double kDeg2Rad = M_PI / 180.0;

struct Camera {
    double fx = 520;
    double cx = 320;
    double fy = 520;
    double cy = 255;
    int width = 640;
    int height = 480;
};

typedef Eigen::Matrix<double, 3, Eigen::Dynamic> DynamicPointMatrix;

DynamicPointMatrix GenerateMapPoints(
    const Camera& cam, const Eigen::Matrix<double, 2, Eigen::Dynamic>& Px);

void AddNoise2Pw(const double addNoiseNumRatio, DynamicPointMatrix& Pw);

Sim3Pose GenerateSim3Pose(const double r, const double p, const double y,
                          const Eigen::Vector3d& Pwc, const double s);

DynamicPointMatrix TransformPw2Pc(const Sim3Pose& sTwc,
                                  const DynamicPointMatrix& Pw);

bool CalculateSim3PosesT12RANSAC(const DynamicPointMatrix& Pc1,
                                 const DynamicPointMatrix& Pc2, Sim3Pose& sT12,
                                 const int minSet = 3,
                                 const double prob = 0.999,
                                 const double inerProb = 0.5);

bool CalculateSim3PoseT12(const DynamicPointMatrix& Pc1,
                          const DynamicPointMatrix& Pc2, Sim3Pose& sT12);

int CalculateInnerNum(const DynamicPointMatrix& Pc1,
                      const DynamicPointMatrix& Pc2, const Sim3Pose& sT12,
                      const double diffRatio);

int main() {
    Camera cam;
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
    const Camera& cam, const Eigen::Matrix<double, 2, Eigen::Dynamic>& Px) {
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

bool CalculateSim3PoseT12(const DynamicPointMatrix& Pc1,
                          const DynamicPointMatrix& Pc2, Sim3Pose& sT12) {

    // 直接调用Eigen库实现
    Eigen::Matrix4d transformT12 = Eigen::umeyama(Pc2, Pc1, true);
    // 方法1：通过行列式计算尺度（3D情况）
    // 因为 det(sR) = s^3 * det(R) = s^3 （det(R)=1）
    const double scale = cbrt(transformT12.determinant());
    transformT12.block<3, 3>(0, 0) /= scale;
    sT12 = Sim3Pose(Eigen::Quaterniond(transformT12.block<3, 3>(0, 0)),
                    transformT12.block<3, 1>(0, 3), scale);

    // debug信息
    // DynamicPointMatrix Pc1FromPc2 = Pc2;
    // for (int j = 0; j < Pc2.cols(); ++j) {
    //     Pc1FromPc2.col(j) = sT12 * Pc2.col(j);
    // }
    // cout << "Pc1:\n" << Pc1 << endl;
    // cout << "Pc1FromPc2:\n" << Pc1FromPc2 << endl;

    /*
    // 1. 去点集中心
    const Eigen::Vector3d c1 = Pc1.rowwise().sum() / 3;
    const Eigen::Vector3d c2 = Pc2.rowwise().sum() / 3;

    // cout << "Pc1:\n" << Pc1 << "\n center1: " << c1.transpose() << endl;
    const DynamicPointMatrix mPc1 = Pc1.colwise() - c1;
    const DynamicPointMatrix mPc2 = Pc2.colwise() - c2;

    // 2. 计算尺度变化
    // pc1 = s * R12 * pc2 + t12
    double scale = 0;
    for (int j = 1; j < Pc1.cols(); ++j) {
        double dist1 = (Pc1.col(j) - Pc1.col(j - 1)).norm();
        double dist2 = (Pc2.col(j) - Pc1.col(j - 1)).norm();
        scale += dist1 / dist2;
    }
    scale /= (Pc1.cols() - 1);

    // 3. 应用尺度变化
    const DynamicPointMatrix sPc2 = scale * Pc2;  // 应用尺度

    // 4. 计算旋转，使用Umeyama方法
    // 去完中心后，相当于把点集放到了原点，两点集之间差一个旋转
    // Pc1 = s * R12 * Pc2
*/
    return true;
}

int CalculateInnerNum(const DynamicPointMatrix& Pc1,
                      const DynamicPointMatrix& Pc2, const Sim3Pose& sT12,
                      const double diffRatio) {
    int innerNum = 0;
    for (int j = 0; j < Pc1.cols(); ++j) {
        const Eigen::Vector3d pc1 = sT12 * Pc2.col(j);
        const double diff = (Pc1.col(j) - pc1).norm();
        if (diff / Pc1.col(j).norm() < diffRatio) {
            ++innerNum;
        }
    }

    return innerNum;
}

bool CalculateSim3PosesT12RANSAC(const DynamicPointMatrix& Pc1,
                                 const DynamicPointMatrix& Pc2, Sim3Pose& sT12,
                                 const int minSet, const double prob,
                                 const double inerProb) {
    if (Pc1.size() != Pc2.size() || Pc1.cols() != Pc2.cols() ||
        Pc1.cols() < minSet) {
        return false;
    }
    // 1. 计算需迭代次数:
    // 1.0 - fail^n > prob
    const double testOneTimeSucceedProb = pow(inerProb, minSet);
    const double testOneTimeFailProb = 1.0 - testOneTimeSucceedProb;
    const int testTime = int(log(1.0 - prob) / log(testOneTimeFailProb) + 0.5);
    cout << "RANSAC need test time: " << testTime << endl;

    mt19937 rng;
    uniform_int_distribution<int> dist(0, Pc1.cols() - 1);
    auto RandomSelectMinSet = [&minSet, &rng, &dist]() -> vector<int> {
        set<int> selected;
        while (selected.size() < minSet) {
            selected.insert(dist(rng));
        }
        return vector<int>(selected.begin(), selected.end());
    };

    // pc1 = s * R12 * pc2 + t12
    int maxInner = 0;
    for (int i = 0; i < testTime; ++i) {
        vector<int> colIndex = RandomSelectMinSet();
        DynamicPointMatrix samplePc1(3, minSet);
        DynamicPointMatrix samplePc2(3, minSet);
        for (int j = 0; j < colIndex.size(); ++j) {
            samplePc1.col(j) = Pc1.col(colIndex[j]);
            samplePc2.col(j) = Pc2.col(colIndex[j]);
        }

        Sim3Pose sT12Temp;
        CalculateSim3PoseT12(samplePc1, samplePc2, sT12Temp);
        const int innerNum = CalculateInnerNum(Pc1, Pc2, sT12Temp, 0.1);
        cout << fmt::format(
                    "Ransac ite: {}th, innerNum: {}, maxInner: {}, total Point "
                    "num: {}",
                    i, innerNum, maxInner, Pc1.cols())
             << endl;
        if (innerNum > maxInner) {
            sT12 = sT12Temp;
            maxInner = innerNum;
        }
    }

    return maxInner > Pc1.cols() * inerProb - 1;
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
