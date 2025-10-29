#include "Utils.h"
#include <unistd.h>
#include "Landmark.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <numeric>  // 用于 std::accumulate

using namespace cv;
using namespace std;

InteractionParam* interaction = nullptr;

std::map<int, cv::Vec3b> Color;
void InitColor() {
    Color.insert({COLOR::red, cv::Vec3b(0, 0, 255)});
    Color.insert({COLOR::orange, cv::Vec3b(0, 165, 255)});
    Color.insert({COLOR::yellow, cv::Vec3b(0, 255, 255)});
    Color.insert({COLOR::green, cv::Vec3b(0, 255, 0)});
    Color.insert({COLOR::blue, cv::Vec3b(255, 255, 0)});
    Color.insert({COLOR::purple, cv::Vec3b(128, 0, 128)});
    Color.insert({COLOR::pink, cv::Vec3b(203, 192, 255)});
}

Mat GetDistanceTransform(Mat img) {
    Mat res;
    // https://blog.csdn.net/kakiebu/article/details/82967085
    distanceTransform(img, res, DIST_L2, 0);
    Assert(res.type() == CV_32FC1, "Assert distance transform type error!");
    // TODO: 是否需要归一化呢，不用
    // cv::normalize(res, res, 0, 1);

    // 归一化显示距离变换结果
    // cv::Mat showDist;
    // cv::normalize(res, showDist, 0, 255, cv::NORM_MINMAX);
    // showDist.convertTo(showDist, CV_8U);
    // cv::imshow("edge", img);
    // cv::imshow("dist", showDist);
    // cv::waitKey(0);
    return res;
}

vector<Eigen::Vector3d> TransformPoint2Pc(const Pose& T,
                                          vector<Eigen::Vector3d>& ps) {
    vector<Eigen::Vector3d> pc;
    for (const Eigen::Vector3d& p : ps) {
        const Eigen::Vector3d t = T * p;
        Assert(t[2] > 0, "[ERROR] depth can't be negative");
        pc.push_back(t);
    }
    return pc;
}

Eigen::Matrix3d SkewSymmetric(const Eigen::Vector3d& v) {
    Eigen::Matrix3d m;
    m.setZero();
    m << 0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0;
    return m;
}

void Assert(bool a, const string& s) {
    if (!a) {
        cerr << s << endl;
        exit(-1);
    }
}

void ShowImage(const cv::Mat& img, const string& name, const bool show) {
    if (!show) {
        return;
    }
    Mat m = img.clone();

    m.convertTo(m, CV_32F);
    cv::normalize(m, m, 255, 0, cv::NORM_MINMAX);
    m.convertTo(m, CV_8UC1);
    cv::imshow(name, m);
    cv::waitKey(0);
}

Eigen::Vector3d Quat2RPY(const Eigen::Quaterniond& _q) {
    const Eigen::Quaterniond q = _q.normalized();
    const double x = q.x(), y = q.y(), z = q.z(), w = q.w();

    // 防止除以零
    double epsilon = 1e-6;

    // roll (x-axis rotation)
    double sinr_cosp = 2.0 * (w * x + y * z);
    double cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
    const double roll = atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis rotation)
    double sinp = 2.0 * (w * y - z * x);
    double pitch = 0;
    if (abs(sinp) >= 1)
        pitch = copysign(M_PI / 2, sinp);  // 使用90度或-90度
    else
        pitch = asin(sinp);

    // yaw (z-axis rotation)
    double siny_cosp = 2.0 * (w * z + x * y);
    double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
    const double yaw = atan2(siny_cosp, cosy_cosp);
    return {roll, pitch, yaw};
}

ostream& operator<<(ostream& cout, const Pose& T) {
    cout << setprecision(5)
         << "RPY | t: " << Quat2RPY(T.q_wb_).transpose() * kRad2Deg << " deg"
         << " | " << T.t_wb_.transpose() * 1000 << " mm";
    return cout;
}

vector<double> CalculateDescriptor(const cv::Mat& grayImg,
                                   const Eigen::Vector2d& px,
                                   const Eigen::Vector2d& epNorm,
                                   const int len) {
    vector<double> des(len, 0.);
    if (len % 2 == 0) {
        cerr << "descriptor length must be odd number" << endl;
        exit(-1);
    }

    if (1) {
        const int mid = len / 2;  // default = 2
        // 这里我们使用双线性插值来获取光度，这样就不用担心四舍五入的问题了
        des[mid] = BilinearInterpolate<uchar>(grayImg, px);
        int incRatio = 1;
        const int maxId = len - 1;  // default 4
        for (int i = mid - 1; i >= 0; --i) {
            des[i] =
                BilinearInterpolate<uchar>(grayImg,
                                           px - incRatio * epNorm);  // 1, 0
            des[maxId - i] = BilinearInterpolate<uchar>(
                grayImg, px + incRatio * epNorm);  // 3, 4
            ++incRatio;
        }
    } else {
        des[0] = BilinearInterpolate<uchar>(grayImg, px + 2 * epNorm);
        des[1] = BilinearInterpolate<uchar>(grayImg, px + 1 * epNorm);
        des[2] = BilinearInterpolate<uchar>(grayImg, px);
        des[3] = BilinearInterpolate<uchar>(grayImg, px - epNorm);
        des[4] = BilinearInterpolate<uchar>(grayImg, px - 2 * epNorm);
    }
    return des;
}

Mat DrawMatch(const Mat& img1, const Mat& img2,
              const vector<Eigen::Vector2i>& kp1,
              const vector<Eigen::Vector2i>& kp2, const string& name,
              const int ratio, const int jump) {
    if (kp1.empty() || kp2.empty()) {
        cerr << "{" + name << "} Error!" << endl
             << "kp1 & kp2 size: " << kp1.size() << " & " << kp2.size() << endl;
        exit(-1);
    }
    //Assert(kp1.size()==kp2.size() || kp1.size() == 1 || kp1.size() < kp2.size(), "match point size error!");
    Mat im(max(img1.rows, img2.rows), (img1.cols + img2.cols), CV_8UC3,
           cv::Scalar{0, 0, 0});
    Mat im1, im2;
    cvtColor(img1, im1, COLOR_GRAY2BGR);
    cvtColor(img2, im2, COLOR_GRAY2BGR);
    im1.copyTo(im.colRange(0, img1.cols));
    im2.copyTo(im.colRange(img1.cols, im.cols));
    cv::resize(im, im, cv::Size(ratio * im.cols, ratio * im.rows));

    const int sCol = img1.cols * ratio;
    cv::Scalar pColor = cv::Scalar(0, 0, 255);
    int radius = 1;
    cv::Scalar lColor = cv::Scalar(0, 255, 0);
    cv::Scalar startColor = cv::Scalar(255, 255, 255);
    cv::Scalar endColor = cv::Scalar(0, 255, 255);
    cv::Scalar matchColor = cv::Scalar(255, 0, 255);

    // 查看 KF2上(x, y)是否与KF1上(x, y)在同一极平面上
    // 实践证明，不会
    // cv::Point matchPxKF1 = cv::Point(kp1[0].x()*ratio, kp1[0].y()*ratio);
    // cv::Point pxKF1inKF2 = cv::Point (sCol+kp1[0].x()*ratio, kp1[0].y()*ratio);
    // cv::line(im, matchPxKF1, pxKF1inKF2, lColor, 1);

    const int maxSize = max(kp2.size(), kp1.size());
    for (int i = 0; i < maxSize; ++i) {
        if (!InRange(img2, kp2[i].cast<int>())) {
            continue;
        }

        cv::Point c1(kp1[0].x() * ratio, kp1[0].y() * ratio);
        cv::Point c2(sCol + kp2[0].x() * ratio, kp2[0].y() * ratio);

        if (i < kp1.size()) {
            c1 = cv::Point(kp1[i].x() * ratio, kp1[i].y() * ratio);
        }

        if (i < kp2.size()) {
            c2 = cv::Point(sCol + kp2[i].x() * ratio, kp2[i].y() * ratio);
        }
        // if((kp1.size() == 1 || 1) && i%10 == 0) {
        //if( i%jump == 0 ) {
        //    cv::line(im, c1, c2, lColor, 1);
        //}
        cv::circle(im, c1, radius, pColor, 1);
        cv::circle(im, c2, radius, pColor, 1);

        if (i == int(kp1.size() - 1)) {
            cv::circle(im, c1, radius * 2, endColor, 2);
            cout << "draw end p1" << kp1.back().transpose() << endl;
            ;
        }
        if (i == int(kp2.size() - 1)) {
            cv::circle(im, c2, radius * 2, endColor, 2);
        }

        if (i == 0) {
            cv::circle(im, c1, radius * 2, startColor, 2);
            cv::circle(im, c2, radius * 2, startColor, 2);
        }
    }

    cv::Point matchPxKF1 =
        cv::Point(kp1.back().x() * ratio, kp1.back().y() * ratio);
    cv::Point pxKF1inKF2 =
        cv::Point(sCol + kp2.back().x() * ratio, kp2.back().y() * ratio);
    cv::line(im, matchPxKF1, pxKF1inKF2, lColor, 1);

    cv::namedWindow(name);
    cv::imshow(name, im);
    cv::imwrite(name + ".png", im);
    cv::waitKey(0);
    return im;
}

char DrawMatch(const cv::Mat& img1, const cv::Mat& img2,
               const std::vector<Eigen::Vector2i>& trajKp1,
               const std::vector<Eigen::Vector2i>& trajKp2,
               const std::vector<Eigen::Vector2i>& goodKp2,
               const std::string& name, const int ratio, const int jump) {
    if (trajKp1.empty() || trajKp2.empty()) {
        cerr << "{" + name << "} Error!" << endl
             << "kp1 & kp2 size: " << trajKp1.size() << " & " << trajKp2.size()
             << endl;
        exit(-1);
    }
    Mat im(max(img1.rows, img2.rows), (img1.cols + img2.cols), CV_8UC3,
           cv::Scalar{0, 0, 0});
    Mat im1, im2;
    cvtColor(img1, im1, COLOR_GRAY2BGR);
    cvtColor(img2, im2, COLOR_GRAY2BGR);
    im1.copyTo(im.colRange(0, img1.cols));
    im2.copyTo(im.colRange(img1.cols, im.cols));
    cv::resize(im, im, cv::Size(ratio * im.cols, ratio * im.rows));

    const int sCol = img1.cols * ratio;
    cv::Scalar pColor = cv::Scalar(255, 0, 0);
    int radius = 1;
    cv::Scalar lColor = cv::Scalar(0, 255, 0);
    cv::Scalar startColor = cv::Scalar(0, 0, 255);
    cv::Scalar bestMatchColor = cv::Scalar(0, 255, 255);
    cv::Scalar secondMatchColor = cv::Scalar(0, 255, 0);

    cv::Point c1(trajKp1[0].x() * ratio, trajKp1[0].y() * ratio);
    cv::circle(im, c1, radius, startColor, 1);
    for (int i = 1; i < trajKp1.size(); ++i) {
        cv::Point c1(trajKp1[i].x() * ratio, trajKp1[i].y() * ratio);
        // cv::circle(im, c1, radius, pColor, 1);
    }

    cv::Point c2(sCol + trajKp2[0].x() * ratio, trajKp2[0].y() * ratio);
    cv::circle(im, c2, radius, startColor, 1);
    for (int i = 1; i < trajKp2.size(); ++i) {
        cv::Point c2(sCol + trajKp2[i].x() * ratio, trajKp2[i].y() * ratio);
        // cv::circle(im, c2, radius, pColor, 1);
    }

    if (goodKp2.size() > 0) {
        cv::Point bestMatchP2 =
            cv::Point(sCol + goodKp2[0].x() * ratio, goodKp2[0].y() * ratio);
        cv::circle(im, bestMatchP2, radius, bestMatchColor, 1);
        //cv::line(im, c1, bestMatchP2, lColor, 1);
    }
    if (goodKp2.size() > 1) {
        cv::Point goodMatchP2 =
            cv::Point(sCol + goodKp2[1].x() * ratio, goodKp2[1].y() * ratio);
        cv::circle(im, goodMatchP2, radius, secondMatchColor, 1);
    }

    cv::Point epipolarPoint1(trajKp1.back().x() * ratio,
                             trajKp1.back().y() * ratio);
    cv::circle(im, epipolarPoint1, radius, bestMatchColor, 1);

    cv::namedWindow(name);
    cv::imshow(name, im);
    cv::imwrite(name + ".png", im);
    return cv::waitKey(0);
}

// 比像素平面上的极线约束多了N次投影到像素平面的运算
vector<Eigen::Vector2d> FindMatchesWithEpipolarConstraintOnImagePlane(
    const Eigen::Vector2d& kp1, const Mat& edgeImg, const Pose& T21,
    const Camera& cam) {
    /******** 使用极线约束寻找匹配关键点 ********
    * R21 * s1 * Pc1_norm + t21 = s2 * Pc2_norm
    * R21 * s1/s2 * Pc1_norm + 1/s2 * t21 = Pc2_norm
    * [t21]x * R21 * s1/s2 * Pc1_norm = [t21]x * Pc2_norm
    * Pc2_norm.T * [t21]x * R21 * s1/s2 * Pc1_norm = 0
    * Pc2_norm.T * [t21]x * R21 * Pc1_norm = 0 --------> 归一化平面上极线约束
    * fx*x+cx = px_x ==> 归一化平面上[1/fx]米对应1个像素
    * fy*y+cy = px_y 
    ********************************************/
    Eigen::Vector3d c = SkewSymmetric(T21.t_wb_) *
                        T21.q_wb_.toRotationMatrix() * cam.InverseProject(kp1);
    // c[0]*x + c[1]*y + c[2] = 0 ==> 归一化平面上的极线
    // y = -c[0]/c[1]*x - c[2]/c[1]

    auto Kp2Useful = [&edgeImg](const Eigen::Vector2i& p) -> bool {
        const int x = p[0], y = p[1];
        // InRange函数保证(y, x)点的十字架处像素不会超过索引
        return InRange(edgeImg, p) &&
               (!edgeImg.at<uchar>(y, x) || !edgeImg.at<uchar>(y - 1, x) ||
                !edgeImg.at<uchar>(y + 1, x) || !edgeImg.at<uchar>(y, x - 1) ||
                !edgeImg.at<uchar>(y, x + 1));
    };

    vector<Eigen::Vector2d> kp2;
    auto IsZero = [](const double a) -> bool {
        return abs(a) < 1e-10;
    };

    const double xStep = 1.0 / cam.fx_, yStep = 1.0 / cam.fy_;
    const double xStart = -cam.cx_ / cam.fx_ + xStep,
                 yStart = -cam.cy_ / cam.fy_ + yStep;
    const double xEnd = -xStart, yEnd = -yStart;

    if (IsZero(c[1]) && IsZero(c[0])) {
        return kp2;
    } else if (IsZero(c[1]) && !IsZero(c[0])) {
        const double x = -c[2] / c[0];  // 归一化平面坐标
        cout << "x: " << x << endl;
        for (double y = yStart; y < yEnd; y += yStep) {
            const Eigen::Vector2d px = cam.Project2PixelPlane({x, y, 1});
            if (Kp2Useful(px.cast<int>())) {
                kp2.push_back(px.cast<double>());
            }
        }
    } else if (!IsZero(c[1]) && IsZero(c[0])) {
        const double y = -c[2] / c[1];
        for (double x = xStart; x < xEnd; x += xStep) {
            const Eigen::Vector2d px = cam.Project2PixelPlane({x, y, 1});
            if (Kp2Useful(px.cast<int>())) {
                kp2.push_back(px.cast<double>());
            }
        }
    } else {
        // y = ax + b
        const double a = -c[0] / c[1], b = -c[2] / c[1];
        for (double x = xStart; x < xEnd; x += xStep) {
            const double y = a * x + b;
            const Eigen::Vector2d px = cam.Project2PixelPlane({x, y, 1});
            if (Kp2Useful(px.cast<int>())) {
                kp2.push_back(px.cast<double>());
            }
        }
    }
    return kp2;
}

Eigen::Vector3d Triangulate(const Eigen::Vector2d& kp2, const Pose& T21,
                            const Camera& cam) {
    /******** 三角化地图点 ********
    * R21 * Pc1 + t21 = z2 * kp2_norm
    * [kp2_norm]x * R21 * Pc1 = -[kp2_norm]x * t21
    *****************************/
    // TODO: 检验为什么该种三角化方式不行！！！
    // 直观理解就是，与kp2出发射线有交点的地图点均可满足该约束，因为没有用到kp1信息，故而无法确定Pc1
    const Eigen::Vector3d kp2Norm = cam.InverseProject(kp2);
    const Eigen::Matrix3d skew = SkewSymmetric(kp2Norm);
    const Eigen::Matrix3d A = skew * T21.q_wb_.toRotationMatrix();
    const Eigen::Vector3d b = -skew * T21.t_wb_;
    return A.colPivHouseholderQr().solve(b);
    //return A.inverse() * b;
}

Eigen::Vector3d Triangulate(const Eigen::Vector2d& kp1,
                            const Eigen::Vector2d& kp2, const Pose& T21,
                            const Camera& cam) {
    /******** 三角化地图点 Pc1 ********
    * R1w * Pw + t1w = z * kp1_norm
    * [kp1_norm]x * R1w * Pw = -[kp1_norm]x * t1w
    *
    * R21 * Pc1 + t21 = z * kp2_norm
    *****************************/
    const Eigen::Vector3d kp1Norm = cam.InverseProject(kp1);
    const Eigen::Vector3d kp2Norm = cam.InverseProject(kp2);
    const Eigen::Matrix3d skew1 = SkewSymmetric(kp1Norm);
    const Eigen::Matrix3d skew2 = SkewSymmetric(kp2Norm);
    Eigen::Matrix<double, 6, 3> A;
    A.block(0, 0, 3, 3) = skew1 * Eigen::Matrix3d::Identity();
    A.block(3, 0, 3, 3) = skew2 * T21.q_wb_.toRotationMatrix();
    Eigen::Matrix<double, 6, 1> b;
    b.middleRows(0, 3) = -skew1 * Eigen::Vector3d::Zero();
    b.middleRows(3, 3) = -skew2 * T21.t_wb_;

    // Eigen::JacobiSVD<Eigen::Matrix<double, 6, 3> > svd(A, Eigen::ComputeFullV);
    // cout << "conditional num: " << svd.singularValues()[0] / svd.singularValues()[2] << endl;
    return A.colPivHouseholderQr().solve(b);
}

double TriangulateDepth(const Eigen::Vector2d& kp1, const Eigen::Vector2d& kp2,
                        const Pose& T21, const Camera& cam) {
    Pose T12 = T21.Inverse();

    Eigen::Vector3d f_ref = cam.InverseProject(kp1);
    f_ref.normalize();
    Eigen::Vector3d f_curr = cam.InverseProject(kp2);
    f_curr.normalize();

    // 方程
    // d_ref * f_ref = d_cur * ( R_RC * f_cur ) + t_RC
    // => [ f_ref^T f_ref, -f_ref^T f_cur ] [d_ref] = [f_ref^T t]
    //    [ f_cur^T f_ref, -f_cur^T f_cur ] [d_cur] = [f_cur^T t]
    // 二阶方程用克莱默法则求解并解之
    Eigen::Vector3d t = T12.t_wb_;
    Eigen::Vector3d f2 = T12.q_wb_ * f_curr;
    Eigen::Vector2d b = Eigen::Vector2d(t.dot(f_ref), t.dot(f2));
    double A[4];
    A[0] = f_ref.dot(f_ref);
    A[2] = f_ref.dot(f2);
    A[1] = -A[2];
    A[3] = -f2.dot(f2);
    double d = A[0] * A[3] - A[1] * A[2];
    Eigen::Vector2d lambdavec =
        Eigen::Vector2d(A[3] * b(0, 0) - A[1] * b(1, 0),
                        -A[2] * b(0, 0) + A[0] * b(1, 0)) /
        d;
    Eigen::Vector3d xm = lambdavec(0, 0) * f_ref;
    Eigen::Vector3d xn = t + lambdavec(1, 0) * f2;
    Eigen::Vector3d d_esti = (xm + xn) / 2.0;  // 三角化算得的深度向量
    double depth_estimation = d_esti.norm();   // 深度值
    return depth_estimation;
}

Pose ConvertRPYandPostion2Pose(const Eigen::Vector3d& rpy,
                               const Eigen::Vector3d& t, const double deg2rad) {
    Eigen::Quaterniond q_c1c2 =
        Eigen::AngleAxisd(rpy[2] * deg2rad, Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(rpy[1] * deg2rad, Eigen::Vector3d::UnitX()) *
        Eigen::AngleAxisd(rpy[0] * deg2rad, Eigen::Vector3d::UnitZ());
    return Pose(q_c1c2, t);
}

void varifyTriangulate() {
    Pose Twc1(Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());
    Pose Tc1c2 =
        ConvertRPYandPostion2Pose({0, 0, 20}, {0.5, 1., -0.5}, kDeg2Rad);
    shared_ptr<Camera> cam = make_shared<Camera>(config);

    const Eigen::Vector2d px1{15, 17};
    const double z1 = 5.0;
    const Eigen::Vector3d pc1 = cam->InverseProject(px1, z1);
    cout << "pc1: " << pc1.transpose() << endl;
    cout << "px1: " << cam->Project2PixelPlane(pc1).transpose() << endl;

    const Eigen::Vector3d pc2 = Tc1c2.Inverse() * pc1;
    cout << "pc2: " << pc2.transpose() << endl;
    const Eigen::Vector2d px2 = cam->Project2PixelPlane(pc2);
    cout << "px2: " << px2.transpose() << endl;

    // 验证像素偏差对三角化精度的影响
    vector<Eigen::Vector2d> px2_9;
    for (int x = -1; x < 2; ++x) {
        for (int y = -1; y < 2; ++y) {
            Eigen::Vector2d px{px2.x() + x, px2.y() + y};
            px2_9.push_back(px);
        }
    }

    for (int i = 0; i < 9; ++i) {
        Eigen::Vector3d pc1_est =
            Triangulate(px1, px2_9[i], Tc1c2.Inverse(), *cam);
        cout << "pc1_est: " << pc1_est.transpose() << endl;
    }
    // 结论：在[3x3]邻域范围内，对三角化精度的影响尚可接受
}

size_t LoadImages(const string& strDirectory, vector<string>& vstrImages,
                  vector<double>& vTimeStamps, const std::string& imgSuffix,
                  const bool readDepth) {
    string imageDirectory = strDirectory + "/image";
    string imageTimestampFile = strDirectory + "/image_timestamp.csv";
    if (readDepth) {
        imageDirectory = strDirectory + "/depth_image";
        imageTimestampFile = strDirectory + "/depth_image_timestamp.csv";
    }

    ifstream fImgTimestamp;
    fImgTimestamp.open(imageTimestampFile.c_str());
    if (!fImgTimestamp.is_open()) {
        cout << "Get image timestamp file failed! Abort!!!" << endl;
        cout << "image name = " << imageTimestampFile << endl;
        abort();
    }

    string input;
    while (getline(fImgTimestamp, input)) {
        if (input.empty()) {
            continue;
        }
        vTimeStamps.push_back(stod(input));
        vstrImages.push_back(imageDirectory + "/" + input + imgSuffix);
    }
    return vstrImages.size();
}

size_t LoadPriorOdom(const string& strDirectory,
                     vector<Eigen::Matrix<double, 8, 1>>& vPriorPose) {
    const string OdomFIle = strDirectory + "/odometry.csv";

    vPriorPose.reserve(20000);

    ifstream fOdom;
    fOdom.open(OdomFIle.c_str());
    if (!fOdom.is_open()) {
        cout << "Get odom failed! Abort!!!" << endl;
        cout << "odom name = " << OdomFIle << endl;
        abort();
    }

    string input;
    while (getline(fOdom, input)) {
        if (input.empty() || input[0] == '#') {
            continue;
        }
        stringstream ss(input);
        double timestamp, px, py, pz, qx, qy, qz, qw;
        ss >> timestamp >> px >> py >> pz >> qx >> qy >> qz >> qw;

        Eigen::Matrix<double, 8, 1> pose;
        pose << timestamp, px, py, pz, qx, qy, qz, qw;
        vPriorPose.push_back(pose);
    }
    // 检查时间戳
    for (int i = 1; i < vPriorPose.size(); ++i) {
        if (vPriorPose[i](0) < vPriorPose[i - 1](0)) {
            cout << "Odom time error! Abort!!!\n";
            abort();
        }
    }
    return vPriorPose.size();
}

void FindImageAndPose(const int idx, const vector<string>& vstrImages,
                      const vector<double> vTimeStamps,
                      const vector<Eigen::Matrix<double, 8, 1>> vPriorPose,
                      const WheelCameraCalib& calib, vector<Mat>& imgs,
                      vector<Pose>& vTwc, const int needNum) {
    Assert(!vstrImages.empty() && !vTimeStamps.empty() && !vPriorPose.empty(),
           "Dataset is empty!!!");

    // 插值pose
    auto InterpolatePose = [&vPriorPose, &calib](const double& t) -> Pose {
        if (t < vPriorPose.front()[0] || t > vPriorPose.back()[0]) {
            cerr << "Can't find match pose!!!" << endl;
            // exit(-1);
        }
        Eigen::Matrix<double, 8, 1> Twv1, Twv2;
        for (int i = 1; i < vPriorPose.size(); ++i) {
            if (vPriorPose[i][0] > t) {
                Twv1 = vPriorPose[i - 1];
                Twv2 = vPriorPose[i];
                break;
            }
        }
        const double ratio = (t - Twv1[0]) / (Twv2[0] - Twv1[0]);
        const Eigen::Vector3d p =
            (1 - ratio) * Twv1.middleRows(1, 3) + ratio * Twv2.middleRows(1, 3);
        const Eigen::Quaterniond q1(Twv1[7], Twv1[4], Twv1[5], Twv1[6]);
        const Eigen::Quaterniond q2(Twv2[7], Twv2[4], Twv2[5], Twv2[6]);
        const Eigen::Quaterniond q = q1.slerp(ratio, q2);
        Pose Twv(q, p);
        return calib.Tcv_ * Twv * calib.Tvc_;
    };

    imgs.push_back(cv::imread(vstrImages[idx], IMREAD_GRAYSCALE));
    const double imgScale = config->imageScale;
    const int newW = imgs[0].cols * imgScale, newH = imgs[0].rows * imgScale;
    cv::resize(imgs[0], imgs[0], cv::Size(newW, newH));
    vTwc.push_back(InterpolatePose(vTimeStamps[idx] + config->imgTimeOffset));

    int id = idx;
    int curId = 1;

    while (id < vstrImages.size() && curId < needNum) {
        // 避免死循环
        ++id;

        const double time = vTimeStamps[id];
        Pose Twc_i = InterpolatePose(time);
        if ((Twc_i.Inverse() * vTwc[curId - 1]).t_wb_.norm() >
            config->minTranslation) {
            imgs.push_back(cv::imread(vstrImages[id], IMREAD_GRAYSCALE));
            cv::resize(imgs[curId], imgs[curId], cv::Size(newW, newH));
            vTwc.push_back(Twc_i);
            ++curId;
        }
    }
    Assert(imgs.size() == needNum && vTwc.size() == needNum,
           "Find imgs and Twc size Error !!!");
}

void GetImageAndPose(const int idx, const vector<string>& vstrImages,
                     const vector<double> vTimeStamps,
                     const vector<Eigen::Matrix<double, 8, 1>> vPriorPose,
                     const WheelCameraCalib& calib, cv::Mat& img, Pose& Twc) {
    Assert(!vstrImages.empty() && !vTimeStamps.empty() && !vPriorPose.empty(),
           "Dataset is empty!!!");

    // 插值pose
    auto InterpolatePose = [&vPriorPose, &calib](const double& t) -> Pose {
        if (t < vPriorPose.front()[0] || t > vPriorPose.back()[0]) {
            cerr << "Can't find match pose!!!" << endl;
            // exit(-1);
        }
        Eigen::Matrix<double, 8, 1> Twv1, Twv2;
        for (int i = 1; i < vPriorPose.size(); ++i) {
            if (vPriorPose[i][0] > t) {
                Twv1 = vPriorPose[i - 1];
                Twv2 = vPriorPose[i];
                break;
            }
        }
        const double ratio = (t - Twv1[0]) / (Twv2[0] - Twv1[0]);
        const Eigen::Vector3d p =
            (1 - ratio) * Twv1.middleRows(1, 3) + ratio * Twv2.middleRows(1, 3);
        const Eigen::Quaterniond q1(Twv1[7], Twv1[4], Twv1[5], Twv1[6]);
        const Eigen::Quaterniond q2(Twv2[7], Twv2[4], Twv2[5], Twv2[6]);
        const Eigen::Quaterniond q = q1.slerp(ratio, q2);
        Pose Twv(q, p);
        return calib.Tcv_ * Twv * calib.Tvc_;
    };

    img = cv::imread(vstrImages[idx], IMREAD_GRAYSCALE);
    // cv::imshow("src gray", img);
    const double imgScale = config->imageScale;
    const int newW = img.cols * imgScale, newH = img.rows * imgScale;
    cv::resize(img, img, cv::Size(newW, newH));
    Twc = InterpolatePose(vTimeStamps[idx] + config->imgTimeOffset);
}

bool GetDepthImage(const double rgbTime,
                   const std::vector<std::string>& vstrImages,
                   const std::vector<double> vTimeStamps, cv::Mat& depth) {
    // 保证O(1)复杂度
    static int id = 1;
    if (rgbTime < vTimeStamps[id] || rgbTime > vTimeStamps.back()) {
        return false;
    }
    while (1) {
        if (rgbTime >= vTimeStamps[id - 1] && rgbTime <= vTimeStamps[id]) {
            depth =
                imread(vstrImages[id], IMREAD_UNCHANGED);  // 不能使用 CV_16sc1
            // cv::imshow("depth", depth);
            // cv::waitKey();
            --id;
            return true;
        }
        ++id;
    }
}

double CalculateScore(const Eigen::Matrix<float, kDescriptorPatchSize, 1>& d1,
                      const Eigen::Matrix<float, kDescriptorPatchSize, 1>& d2) {
    /****************
    * +---+---+---+
    * + 1 + 2 + 1 +
    * +---+---+---+
    * + 2 + 3 + 2 +
    * +---+---+---+
    * + 1 + 2 + 1 +
    * +---+---+---+
    *****************/
    constexpr double ratio = 1.0 / 15;
    const Eigen::Matrix<float, kDescriptorPatchSize, 1> d =
        (d1 - d2).cwiseAbs();
    const double cost = d[0] + 2 * d[1] + d[2] + 2 * d[3] + 3 * d[4] +
                        2 * d[5] + d[6] + 2 * d[7] + d[8];
    // return (cost/15)/9;
    return cost * ratio;
}

uint64_t CalculateDescriptor(const Mat& grayImg, const Eigen::Vector2i& px) {
    // 返回descDim维描述子, [8x8]的范围内对角线位置的像素值比值
    uint64_t des = 0;
    const int ox = px.x(), oy = px.y(), sx = -4, ex = 4, sy = -4, ey = 3;
    int move = -1;
    for (int i = sx; i <= ex; ++i) {
        for (int j = sy; j <= ey; ++j) {
            const uchar v1 = grayImg.at<uchar>(oy + j, ox + i);
            const uchar v2 = grayImg.at<uchar>(oy - j, ox - i);
            ++move;
            int v = bool(v1 > v2);
            v = v << move;
            des |= v;
        }
    }

    // for(int i = 0; i < 32; ++i) {
    //     int v = 1;
    //     v = v << i;
    //     int b = bool(v&des);
    //     cout << b << " ";
    // }
    // cout << endl;
    return des;
}

double CalculateSSD(const std::vector<double>& v1,
                    const std::vector<double>& v2, double avg1, double avg2,
                    const int desLen) {
    double sum = 0;
    double avg = avg1 - avg2;
    if (!config->useAvgDiff)
        avg = 0;
    for (int i = 0; i < desLen; ++i) {
        sum += abs(v1[i] - avg - v2[i]);
    }
    return sum;
}

double CalculateSSD(const std::vector<double>& v1,
                    const std::vector<double>& v2, const int desLen) {
    double sum = 0;
    double avg = (std::accumulate(v1.begin(), v1.end(), 0.0) -
                  std::accumulate(v2.begin(), v2.end(), 0.0)) /
                 v1.size();
    if (!config->useAvgDiff)
        avg = 0;
    for (int i = 0; i < desLen; ++i) {
        sum += abs(v1[i] - avg - v2[i]);
    }
    return sum;
}

double GetOnePixelUncertainty(const Eigen::Vector3d& t12,
                              const Eigen::Vector3d& pc1, const double f) {
    const Eigen::Vector3d pc2 = pc1 - t12;
    const double pc1Norm = pc1.norm(), t12Norm = t12.norm(),
                 d1 = pc1Norm * t12Norm, d2 = pc2.norm() * t12Norm;
    const double alpha = acos(pc1.dot(t12) / d1);
    const double belta = acos(pc2.dot(-t12) / d2);
#if 0
     const double deltaBelta = atan2(config->filterPixelError, f);
#else
    const double deltaBelta = config->filterPixelError * kDeg2Rad;
#endif

    const double belta2 = belta + deltaBelta;
    const double gamma = M_PI - alpha - belta2;
    const double newPc1Norm = t12Norm * sin(belta2) / sin(gamma);
    {
        static bool first = true;
        ofstream f;
        const string depthUncertainty("get_one_pixel_depth_uncertainty.csv");
        if (first) {
            first = false;
            f.open(depthUncertainty, ios::out);
            f << "#oldDepth, obvDepth, deltaDepth" << endl;
            f.close();
        }
        f.open(depthUncertainty, ios::app);
        f << pc1.z() << " " << newPc1Norm << " " << abs(pc1.z() - newPc1Norm)
          << endl;
        f.close();
    }
    return abs(pc1.z() - newPc1Norm);
}

double GetDepthUncertainty(const Eigen::Vector2d& px2,
                           const Eigen::Vector2d& deltaPix2, const double d,
                           const Camera& cam) {
    // 在极平面上{假设已经进行极线矫正}, 设右视图沿着极线方向产生了Δx个像素偏移，真实值为x2
    // 那么，O2P2{归一化平面上对应x2的点}与O1O2的夹角为：
    // β2 = 1/{(x2-cx)/fx} = fx/(x2-cx)，同理，产生偏移后，有：
    // β2' = fx/(x2-Δx-cx)
    // 设深度方向与O1O2垂足为H，令 b2 = O1O2-O1H
    // 那么，深度d由三角函数计算式有：
    // tanβ2 = d/b2
    // tanβ2' = d'/b2
    // ==> d/d' = tanβ2/tanβ2'
    // Δd = |d - d'| = | (tanβ2/tanβ2' - 1) * d'|
    // 或者 Δd = | (tanβ2'/tanβ2 - 1) * d |
    const Eigen::Vector2d px2_ = px2 + deltaPix2,
                          p2Norm = cam.InverseProject(px2).head(2),
                          p2Norm_ = cam.InverseProject(px2_).head(2);

    const double tanBelta2 = 1.0 / p2Norm.norm();
    const double tanBelta2_ = 1.0 / p2Norm_.norm();  // > 0的实数
    const double deltaDepth = abs(tanBelta2_ / tanBelta2 - 1) * d;
    {
        static bool first = true;
        ofstream f;
        const string depthUncertainty("get_depth_uncertainty.csv");
        if (first) {
            first = false;
            f.open(depthUncertainty, ios::out);
            f << "#deltaPixLen, belta2, delta2_, p2Norm, p2Norm_, depth, "
                 "deltaDepth"
              << endl;
            f.close();
        }
        f.open(depthUncertainty, ios::app);
        f << deltaPix2.norm() << " " << atan(tanBelta2) * kRad2Deg << " "
          << atan(tanBelta2_) * kRad2Deg << " " << p2Norm.norm() << " "
          << p2Norm_.norm() << " " << d << " " << deltaDepth << endl;
        f.close();
    }
    return deltaDepth;
}

bool NeedNewKF(const KeyFrame* kf, const KeyFrame* f) {
    const Pose T12 = kf->priorTwc_.Inverse() * f->priorTwc_;
    return T12.t_wb_.norm() > config->needNewKFtrans ||
           Quat2RPY(T12.q_wb_).norm() * kRad2Deg > config->needNewKFrot;
}

bool IsFastPoint(const cv::Mat& gray, const int fastTh1, const cv::Point2i& pt,
                 int& response) {
    if (config->fastNum == 0) {
        return true;
    }

    constexpr int imgEdgeLen = 6;
    const int maxX = gray.cols - imgEdgeLen;
    const int maxY = gray.rows - imgEdgeLen;
    if (pt.x < imgEdgeLen || pt.x > maxX || pt.y < imgEdgeLen || pt.y > maxY) {
        return false;
    }

    int maxNum = 0;
    int minNum = 0;
    const int v = gray.at<uchar>(pt);

    for (int i = 0; i < 16; ++i) {
        const Point2i pt2{pt.x + FASTpoint[i][0], pt.y + FASTpoint[i][1]};
        if (pt2.x < imgEdgeLen || pt2.x > maxX || pt2.y < imgEdgeLen ||
            pt2.y > maxY) {
            return false;
        }
        const int v2 = gray.at<uchar>(pt2);
        const int diff = v - v2;
        if (diff > fastTh1) {
            ++maxNum;
            response += diff;
        } else if (diff < -fastTh1) {
            ++minNum;
            response -= diff;
        }
    }
    return maxNum > config->fastNum || minNum > config->fastNum;
}

Eigen::Vector3d LogSO3(const Eigen::Matrix3d& R) {
    const double tr = R(0, 0) + R(1, 1) + R(2, 2);
    Eigen::Vector3d w;
    w << (R(2, 1) - R(1, 2)) / 2, (R(0, 2) - R(2, 0)) / 2,
        (R(1, 0) - R(0, 1)) / 2;
    const double costheta = (tr - 1.0) * 0.5f;
    if (costheta > 1 || costheta < -1)
        return w;
    const double theta = acos(costheta);
    const double s = sin(theta);
    if (fabs(s) < 1e-5)
        return w;
    else
        return theta * w / s;
}

Eigen::Matrix3d InverseRightJacobianSO3(const Eigen::Vector3d& v) {
    const double x = v[0], y = v[1], z = v[2];
    const double d2 = x * x + y * y + z * z;
    const double d = sqrt(d2);

    Eigen::Matrix3d W;
    W << 0.0, -z, y, z, 0.0, -x, -y, x, 0.0;
    if (d < 1e-5)
        return Eigen::Matrix3d::Identity();
    else
        return Eigen::Matrix3d::Identity() + W / 2 +
               W * W * (1.0 / d2 - (1.0 + cos(d)) / (2.0 * d * sin(d)));
}

void VizInteraction(const cv::viz::KeyboardEvent& event, void* _b) {
    // 因为q or Q键是默认注册的按键，所以...
    // 如果使用它们，反应有延迟
    if (event.action == viz::KeyboardEvent::KEY_DOWN &&
        (event.code == 'A' || event.code == 'a')) {
        interaction->resetWindow = true;
        interaction->window->setViewerPose(cv::Affine3d::Identity());

    } else if (event.action == viz::KeyboardEvent::KEY_DOWN &&
               (event.code == 'S' || event.code == 's' || event.code == ' ')) {

        cout << "before interaction->stepBystep: " << interaction->stepBystep
             << endl;
        interaction->stepBystep = !interaction->stepBystep;
        cout << "after interaction->stepBystep: " << interaction->stepBystep
             << endl;

    } else if (event.action == viz::KeyboardEvent::KEY_DOWN &&
               (event.code == 'D' || event.code == 'd')) {
        interaction->drawEpipolarMatch = !interaction->drawEpipolarMatch;
    }
}

double CalculatePatchSSD(const KeyFrame* kf1, const KeyFrame* kf2,
                         const Eigen::Vector2i& px1,
                         const Eigen::Vector2i& px2) {
    const Mat& im1 = kf1->grayImg_;
    const Mat& im2 = kf2->grayImg_;
    const int range = config->descriptorPatchLen / 2;
    const int x1 = px1.x(), y1 = px1.y(), x2 = px2.x(), y2 = px2.y();
    double sum = 0;
    double avg = 0;
#if 1
    double sum1 = 0, sum2 = 0;
    int count = 0;
    for (int i = -range; i <= range; ++i) {
        for (int j = -range; j <= range; ++j) {
            sum1 += double(im1.at<uchar>(y1 + i, x1 + j));
            sum2 += double(im2.at<uchar>(y2 + i, x2 + j));
            ++count;
        }
    }
    double avg1 = sum1 / count, avg2 = sum2 / count;
    avg = avg1 - avg2;
#endif

    for (int i = -range; i <= range; ++i) {
        for (int j = -range; j <= range; ++j) {
            sum += abs(double(im1.at<uchar>(y1 + i, x1 + j)) -
                       double(im2.at<uchar>(y2 + i, x2 + j)) - avg);
        }
    }
    return sum / count;
}

double GetPositiveDepth(const double invZ) {
    return invZ < 1e-9 ? 100.0 : 1.0 / invZ;
}

bool GetHostAndCurFrameObservationDepth(const Eigen::Vector2d& kp1,
                                        const Eigen::Vector2d& kp2,
                                        const Eigen::Matrix3d& invK0,
                                        const Pose& T12, double& idepth1,
                                        double& idepth2) {
    // s1 * Pn1 = R12 * s2 * Pn2 + P12
    // [Pn1 - R12*Pn2]_[3x2] * [s1, s2] = P12
    // const Eigen::Vector3d pn1 = invK0 * Eigen::Vector3d(kp1.x(), kp1.y(), 1.0);
    // const Eigen::Vector3d pn2 = invK0 * Eigen::Vector3d(kp2.x(), kp2.y(), 1.0);

    // const Eigen::Matrix3d rot_12 = T12.q_wb_.toRotationMatrix();
    // const Eigen::Vector3d& pos_12 = T12.t_wb_;
    // // 第一种解法
    // Eigen::Matrix<double, 3, 2> matrix_a = Eigen::Matrix<double, 3, 2>::Zero();
    // matrix_a.col(0) = pn1;
    // matrix_a.col(1) = -(rot_12 * pn2);
    // const Eigen::Vector2d res0 =
    //     matrix_a.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV)
    //         .solve(pos_12);

    // depth1 = res0.x();
    // depth2 = res0.y();
    if (!GetHostFrameObservationInvDepth(kp1, kp2, invK0, T12, idepth1)) {
        return false;
    }
    if (!GetHostFrameObservationInvDepth(kp2, kp1, invK0, T12.Inverse(),
                                         idepth2)) {
        return false;
    }

    const double ratio = idepth1 / idepth2;
    if (ratio > 0.75 && ratio < 1.25) {
        return true;
    }
    return false;
}

bool GetHostFrameObservationInvDepth(const Eigen::Vector2d& kp1,
                                     const Eigen::Vector2d& kp2,
                                     const Eigen::Matrix3d& invK0,
                                     const Pose& T12, double& idepth1) {
    // s1 * Pn1 = R12 * s2 * Pn2 + P12
    // s1 * R21 * Pn1 = s2 * Pn2 + R21 * P12
    // s1 * [Pn2]x * R21 * Pn1 = [Pn2]x * R21 * P12
    const Eigen::Vector3d pn1 = invK0 * Eigen::Vector3d(kp1.x(), kp1.y(), 1.0);
    const Eigen::Vector3d pn2 = invK0 * Eigen::Vector3d(kp2.x(), kp2.y(), 1.0);

    const Eigen::Matrix3d rot_21 = T12.q_wb_.toRotationMatrix().transpose();
    const Eigen::Vector3d& pos_12 = T12.t_wb_;
    const Eigen::Matrix3d m = SkewSymmetric(pn2) * rot_21;
    const Eigen::Vector3d p1 = m * pn1;
    const double denominator = (m * pos_12).dot(p1);
    if (abs(denominator) < 1e-9) {
        return false;
    }
    idepth1 = (p1.dot(p1)) / (m * pos_12).dot(p1);

    if (idepth1 < 0. || idepth1 > 1.0 / kMinSceneDepthInCamera) {
        return false;
    }

    // TODO：还要检验视差角
    const Eigen::Vector3d ray1 = -pn1 / idepth1;  // 相机1系下的地图点坐标
    const Eigen::Vector3d ray2 = ray1 + T12.t_wb_;
    const double cosAng = ray1.dot(ray2) / (ray1.norm() * ray2.norm());
    // 必须在3度到120度范围内
    const double ang = acos(cosAng) * kRad2Deg;
    constexpr double kMinCrossAng = 1.0;
    constexpr double kMaxCrossAng = 120.0;
    if (ang < kMinCrossAng || ang > kMaxCrossAng) {
        // cout << "triangulate ang: " << ang << "deg! Error!" << endl;
        return false;
    }

    return true;
}

double CalculateVarianceByOffsetPx2(const Eigen::Vector2d& kp1,
                                    const Eigen::Vector2d& kp2,
                                    const Eigen::Vector2d& ep2,

                                    const Eigen::Matrix3d& invK0,
                                    const Pose& T12, const double offsetRatio) {
    const Eigen::Vector2d pxOffset1 = kp2 - offsetRatio * ep2;
    const Eigen::Vector2d pxOffset2 = kp2 + offsetRatio * ep2;
    double idepth1 = 0;
    double idepth2 = 0;
    if (!GetHostFrameObservationInvDepth(kp1, pxOffset1, invK0, T12, idepth1)) {
        return 1e6;
    }
    if (!GetHostFrameObservationInvDepth(kp1, pxOffset2, invK0, T12, idepth2)) {
        return 1e6;
    }

    return pow(idepth1 - idepth2, 2) * 0.25;  // 取一半
}

double CalculateVariance(const double& estInvDepth1, const Eigen::Vector2d& kp1,
                         const Eigen::Vector2d& kp2, const Pose& T21,
                         const Eigen::Matrix3d& invK,
                         const Eigen::Matrix3d& K) {
    const Eigen::Matrix3d R21 = T21.q_wb_.toRotationMatrix();
    const Eigen::Vector3d& P21 = T21.t_wb_;
    const Eigen::Vector3d Pn1 = invK * Eigen::Vector3d(kp1.x(), kp1.y(), 1.0);

    const Eigen::Vector3d Pc2 = R21 * 1.0 / estInvDepth1 * Pn1 + P21;
    const Eigen::Vector3d Pn2 = Pc2 / Pc2.z();
    const Eigen::Vector2d obv2 = K.block(0, 0, 2, 3) * Pn2;
    const Eigen::Vector2d residual = obv2 - kp2;

    // 求残差关于ρ1的雅可比，据次推导
    // r = J * ρ1
    // J.T * r = J.T * J * ρ1 = a * ρ1
    // ρ1 = 1.0/a * J.T * r
    // r服从N～(0, Σ)高斯分布
    // 令A=1.0/a * J.T， 则ρ1服从N~(ρ1, A*Σ*A.T)
    const Eigen::Vector2d J_res_rho1 =
        CalculateObvWrtIdepth1Jacobian(R21, estInvDepth1, Pn1, Pc2, K);

    // 改进的噪声模型
    const double basePixelNoise = 3.0;  // 基础像素噪声
    const double adaptiveNoise = basePixelNoise + residual.norm();

    const double sigma2 = adaptiveNoise;  // TODO：这里应该加上像素误差比较合理
    const Eigen::Matrix2d obv2SigmaSquare =
        Eigen::Matrix2d::Identity() * sigma2 * sigma2;
    const double h = J_res_rho1.transpose() * J_res_rho1;
    const Eigen::Matrix<double, 1, 2> A = 1.0 / h * J_res_rho1.transpose();
    const double variance = A * obv2SigmaSquare * A.transpose();
    // return ResetVariance(variance); // TODO：是否有必要限制
    return variance;
}

Eigen::Vector2d CalculateObvWrtIdepth1Jacobian(const Eigen::Matrix3d& Rc2_c1,
                                               const double& rho1,
                                               const Eigen::Vector3d& Pn1,
                                               const Eigen::Vector3d& Pc2,
                                               const Eigen::Matrix3d& K) {

    const Eigen::Matrix<double, 2, 3> J_r_Pn2 = K.block(0, 0, 2, 3);
    const double invZ = 1 / Pc2[2];
    const double invZ2 = invZ * invZ;
    // clang-format off
        const Eigen::Matrix3d J_Pn2_Pc2 =
            (Eigen::Matrix3d() << invZ, 0, -Pc2[0] * invZ2, 
                                0, invZ, -Pc2[1] * invZ2,
                                0, 0, 0).finished();
    // clang-format on
    const Eigen::Matrix3d& J_Pc2_Pc1 = Rc2_c1;

    const double invSquareRho1 = 1.0 / (rho1 * rho1);
    const Eigen::Vector3d J_Pc1_rho1 = -invSquareRho1 * Pn1;
    const Eigen::Vector2d J_residual_rho1 =
        J_r_Pn2 * J_Pn2_Pc2 * J_Pc2_Pc1 * J_Pc1_rho1;
    return J_residual_rho1;
}

// 坐标系变换（带协方差传播）
bool LandmarkTransformHost(const Landmark& lk1, const Pose& T21,
                           const Eigen::Matrix3d& invK, const double& depth2,
                           double& variance2, int& varianceDecreaseNum,
                           const double obvResidual) {
    // 1.0/ρ2 * Pn2 = R21 * 1.0/ρ1 * Pn1 + P21
    // 1.0/ρ2 * (Pn2.T * Pn2) = 1.0/ρ1 * (Pn2.T * R21 * Pn1) + (Pn2.T * P21)
    // 1.0/ρ2 * A = 1.0/ρ1 * B + C
    // 1.0/ρ2 = 1.0/ρ1 * B/A + C/A
    // ρ2 = A/(1.0/ρ1 * B + C) = A/t

    // dρ2/dρ1 = -A/t^2 * -B/ρ1^2 = AB/(t*ρ1)^2

    // 正确推导核心：直接在第一步取z分量推导即可
    // 1.0/ρ2 = 1.0/ρ1 * (R21 * Pn1)z + (P21)z
    // ρ2 = 1.0 / (1.0/ρ1 * (R21 * Pn1)z + (P21)z) = 1.0 / (1.0/ρ1 * A + B)

    // dρ2/dρ1 = -1.0/(1.0/ρ1 * A + B)^2 * -A/ρ1^2 = A/(1.0/ρ1 * A * ρ1  + B * ρ1)^2 = A/(A+B*ρ1)^2
    if (depth2 < config->minDepth || depth2 > config->maxDepth) {
        return false;
    }

    const double& idepth1 = lk1.invZ_;
    const double& variance1 = lk1.invDepthCov_;
    const Eigen::Matrix3d Rc2_c1 = T21.q_wb_.toRotationMatrix();
    const Eigen::Vector3d Pn1 =
        invK * Eigen::Vector3d(lk1.uv_.x(), lk1.uv_.y(), 1.0);

    const double J_rho2_d2 = -1.0 / (depth2 * depth2);
    const double J_d2_rho1 = -(Rc2_c1 * Pn1).z() / (idepth1 * idepth1);

    const double J = J_rho2_d2 * J_d2_rho1;
    // TODO：这里应该要考虑基线以设置比率？
    if (obvResidual < sqrt(2.0)) {
        variance2 = J * variance1 * J * 2.0;
    } else {
        variance2 = J * variance1 * J * pow(obvResidual, 2);
    }

    if (variance2 < variance1) {
        cout << fmt::format("Warnning var1:{}>var2:{}!, reset variance2\n",
                            variance1, variance2);
        variance2 = 2.0 * variance1;
        ++varianceDecreaseNum;
    }
    return true;
}

Eigen::Vector2d GetEpipolarLineDirection(const Eigen::Vector3d& Pother2this,
                                         const Eigen::Vector2d& p1,
                                         const Camera& cam) {
    if (abs(Pother2this[2]) < 1e-9) {
        return {0, 0};
    }
    // 返回远点->极点的方向
    // ep1 = (x, y)*t12.z - 极点e1*t12.z
    // 注意：这个极线是像素平面上放大 t12.z 倍后的方向向量
    const double fx = cam.fx_, fy = cam.fy_, cx = cam.cx_, cy = cam.cy_;
    const Eigen::Vector2d ep1{
        -fx * Pother2this[0] + Pother2this[2] * (p1.x() - cx),
        -fy * Pother2this[1] + Pother2this[2] * (p1.y() - cy)};

    return -ep1 / Pother2this[2];
}

Eigen::Vector2i ParseKeypointSet(const std::string& s) {
    // 解析"num_num"为数字
    const int _pos = s.find_first_of('_');
    return {stoi(s.substr(0, _pos)), stoi(s.substr(_pos + 1, s.size()))};
}

bool CheckEpipolarLineDirection(const Eigen::Vector2d& ep2,
                                Eigen::Vector2d& ep1) {
    // 保证极线方向在图像上遵循一致的方向
    const double cosValue = ep1.dot(ep2) / (ep1.norm() * ep2.norm());
    const double ang = acos(cosValue) * kRad2Deg;
    constexpr double kMaxCrossAng = 30;  // deg
    if (!(abs(ang) < kMaxCrossAng || abs(ang) > 180 - kMaxCrossAng)) {
        return false;
    }
    if (cosValue < 0) {
        ep1 *= -1;
    }
    return true;
}

string GetTriangulatePointName(const Eigen::Vector2d& p) {
    return fmt::format("{}_{}", int(p.x() + 0.5), int(p.y() + 0.5));
}

Pose GetPredictPose(const KeyFrame& last1, const KeyFrame& last2) {
    const Pose Tc2c1 = last2.Tcw_ * last1.Twc_;
    const Pose predictCurFpose = last1.Twc_ * Tc2c1;  // 匀速模型

    //cout << "lastLast pose: " << last2.Twc_ << "\nlast pose: " << last1.Twc_
    //     << "\nT21: " << Tc2c1 << "\npredictCurFpose: " << predictCurFpose
    //     << "\n";
    return predictCurFpose;
}

char DrawPerpendicularAndParallelDirectionOFedge(const Mat& edgeImg,
                                                 const Mat& dxImg,
                                                 const cv::Mat& dyImg) {
    // 以一元二次函数为例，(dx, dy)组成了某点处的切线方向，而
    // dx = (x+Δ)-x，y = f(x+Δ)-f(x)，所以：
    // 垂线方向应该要垂直于梯度方向才对???
    // !!!!!!!!!!!!!!!!
    // 但是图像边缘的垂直方向显示就是其梯度(dx, dy)的方向，
    // 边缘的定义就是灰度变化剧烈的地方，
    // 与这里的区别是，一元二次函数的dx是自变量，
    // 而灰度差分这里的dx同dy一样都是因变量
    // 参考讨论： https://zhuanlan.zhihu.com/p/62718992
    // 最终显示：灰度变化的方向就是边缘的法向量
    Mat img;
    cvtColor(edgeImg, img, COLOR_GRAY2BGR);

    uchar* edata = edgeImg.data;
    const float* xdata = dxImg.ptr<float>();
    const float* ydata = dyImg.ptr<float>();
    const int lineLen = 10;
    const Vec3b perColor{0, 255, 0};  // 绿线垂直边缘
    const Vec3b parColor{0, 0, 255};  // 红线平行边缘
    // 跳过一些边缘点
    const int xJump = 10, yJump = 10;
    const int w = img.cols, h = img.rows;
    int count = 0;
    for (int i = 30; i < h - 30; ++i)
        for (int j = 30; j < w - 30; ++j) {
            const int id = i * w + j;
            if (edata[id] == 0) {
                const float dx = xdata[id];
                const float dy = ydata[id];
                const float len = sqrt(dx * dx + dy * dy);
                // 归一化方向，
                // 根据前述及实际显示，图像灰度差分(dx, dy)组成的方向垂直于其边缘像素
                float perDir[2] = {dx / len, dy / len};
                float parDir[2] = {-dy / len, dx / len};

                Point2i o(j, i);
                Point2i per{j + int(perDir[0] * lineLen + 0.5),
                            i + int(perDir[1] * lineLen + 0.5)};
                Point2i par{j + int(parDir[0] * lineLen + 0.5),
                            i + int(parDir[1] * lineLen + 0.5)};

                cv::line(img, o, per, perColor, 1);
                cv::line(img, o, par, parColor, 1);
                ++count;
            }
        }
    cout << "draw " << count << "lines" << endl;
    cv::imshow("edge structure", img);
    return cv::waitKey(0);
}

void ShowPointCloud(const vector<Landmark*>& ps) {
    viz::Viz3d window("One Frame Point Cloud Viewer");
    cv::Affine3d viewPose;
    window.setViewerPose(viewPose);
    vector<Point3d> points;
    vector<Vec3b> colors;

    for (Landmark* _p : ps) {
        const Landmark& p = *_p;
        if (_p == nullptr || !p.Converge()) {
            continue;
        }
        // const double depth = p.depthRange_[0]; // p.z_;//0.5 * (p.depthRange_[0] + p.depthRange_[1]);
        // const Eigen::Vector3d &pc = p.cam_->InverseProject(p.uv_, depth);
        const Eigen::Vector3d& pc = p.GetPw();
        points.push_back({pc.x(), pc.y(), pc.z()});
        // cout << "[" << p.depthRange_[0] << " " << p.depthRange_[1] << "]  ";

        // {B G R}
        if (pc.z() > 8) {
            colors.push_back({255, 255, 0});
        } else if (pc.z() > 4) {
            colors.push_back({0, 255, 0});
        } else if (pc.z() > 2) {
            colors.push_back({0, 0, 255});
        } else {
            colors.push_back({255, 255, 255});
        }
        //cout << pc.z() << ", ";
    }
    cout << endl;
    cout << "show point size: " << points.size() << endl;

    if (points.empty()) {
        cerr << "No Points' depth Converged!" << endl;
        return;
    } else {
        cout << points.size() << " points converged!" << endl;
    }

    // 显示一个长方体点云
    // for(float i = 0; i < 10; i+=0.1) {
    //     for(float j = 0; j < 10; j+=0.1) {
    //         points.push_back({i, j, double(rand()%100)});
    //         colors.push_back({255, 255, 255});
    //     }
    // }

    // 创建点云对象
    viz::WCloud cloud(points, colors);

    // 设置点云颜色和大小
    // cloud.setColor(cv::viz::Color::green());
    // cloud.setSize(5);

    // 显示点云
    window.showWidget("PointCloud", cloud);

    // 运行事件循环，使窗口响应用户输入
    window.spin();
}

void ShowPointCloud(const vector<Landmark*>& ps1, const vector<Landmark*>& ps2,
                    const std::string& windowName, const double zOffset) {
    viz::Viz3d window(windowName);
    cv::Affine3d viewPose;
    window.setViewerPose(viewPose);
    vector<Point3d> points1, points2;

    auto Generate = [](const vector<Landmark*>& ps, vector<Point3d>& points) {
        for (const Landmark* p : ps) {
            if (p == nullptr || !p->Converge()) {
                continue;
            }
            const Eigen::Vector3d& pw = p->GetPw();
            points.push_back({pw.x(), pw.y(), pw.z()});
        }

        if (points.empty()) {
            cerr << "No Points' depth Converged!" << endl;
            return;
        } else {
            cout << points.size() << " points converged!" << endl;
        }
        cout << "show points size: " << points.size() << endl;
    };

    Generate(ps1, points1);
    Generate(ps2, points2);
    vector<Vec3b> colors1(points1.size(), {0, 0, 255}),
        colors2(points2.size(), {0, 255, 0});
    for (auto& p : points1) {
        //p.x += 1.0;
        //p.y += 1.0;
        p.z += zOffset;
    }

    // 创建点云对象
    viz::WCloud cloud1(points1, colors1);
    viz::WCloud cloud2(points2, colors2);
    cloud2.setRenderingProperty(viz::RenderingProperties::POINT_SIZE, 2.);

    // 显示点云
    window.showWidget("PointCloud1", cloud1);
    window.showWidget("PointCloud2", cloud2);

    // 运行事件循环，使窗口响应用户输入
    window.spin();
}

void ShowPointCloud(const set<Landmark*>& ps) {
    if (ps.empty()) {
        return;
    }
    viz::Viz3d window("LocalMap Viewer");
    cv::Affine3d viewPose;
    window.setViewerPose(viewPose);
    vector<Point3d> points;

    for (Landmark* p : ps) {
        if (p == nullptr || !p->Converge()) {
            continue;
        }
        const Eigen::Vector3d pw = p->GetPw();
        points.push_back({pw.x(), pw.y(), pw.z()});
    }
    vector<Vec3b> colors(points.size(), {0, 255, 0});

    // 创建点云对象
    viz::WCloud cloud(points, colors);

    // 显示点云
    window.showWidget("LocalMap", cloud);

    // 运行事件循环，使窗口响应用户输入
    // window.spinOnce(3000);
    window.spin();
}

void ShowLocalMap(const vector<Pose>& vTwc) {

    if (interaction->drawEpipolarMatch ||
        interaction->visualCurF.grayImg_.empty() ||
        interaction->visualCurFinit.grayImg_.empty() ||
        interaction->visualLastKF.grayImg_.empty()) {
        sleep(1);
        return;
    }

    viz::Viz3d& window = *interaction->window;
    //cv::Affine3d &viewPose = *interaction->viewPose;
    //window.setViewerPose(viewPose); // 使用默认的才是正确的
    KeyFrame* curf = &interaction->visualCurF;
    KeyFrame* curfInit = &interaction->visualCurFinit;
    KeyFrame* curkf = &interaction->visualLastKF;
    if (curf->grayImg_.empty()) {
        curf = nullptr;
    }

    Mat curImg, curInitImg;
    const unsigned int curId = curf->id_;
    if (curf != nullptr) {
        cvtColor(curf->debugGrayImg_, curImg, cv::COLOR_GRAY2BGR);
        cvtColor(curfInit->debugGrayImg_, curInitImg, cv::COLOR_GRAY2BGR);
    }
    Mat curKFimg;
    cvtColor(curkf->debugGrayImg_, curKFimg, cv::COLOR_GRAY2BGR);

    // 可视化点云
    auto GenerateCloud = [&curf, &curfInit, &curkf, &curImg, &curInitImg,
                          &curKFimg](set<Landmark*>& ps, const cv::Vec3b& color,
                                     vector<Point3d>& points,
                                     vector<cv::Vec3b>& colors) {
        points.reserve(10000);

        for (Landmark* p : ps) {
            if (p == nullptr || !p->initialized_ || p->CanBeDelete()) {
                continue;
            }
            const Eigen::Vector3d pw = p->GetPw();
            const Eigen::Vector3d pc = p->GetPc();
            points.push_back({pw.x(), pw.y(), pw.z()});
            const int colorId = min(int(pc.z() / 0.5), int(COLOR::pink));
            const cv::Vec3b curColor(Color[static_cast<COLOR>(colorId)]);
            //points.push_back({pc.x(), pc.y(), pc.z()});
            colors.push_back(curColor);

            //if (curf != nullptr && colors.back() == Color[COLOR::red]) {
            if (curf != nullptr) {
                const Eigen::Vector3d pc2 = curf->Tcw_ * pw;
                const Eigen::Vector2i px2 =
                    curf->cam_->Project2PixelPlane(pc2).cast<int>();
                if (InRange(curf->grayImg_, px2)) {
                    //curImg.at<cv::Vec3b>(px2.y(), px2.x()) = {0, 0, 255};
                    // cv::circle(curImg, {px2.x(), px2.y()}, 2, color);
                    cv::circle(curImg, {px2.x(), px2.y()}, 2, colors.back());
                }

                const Eigen::Vector3d pc3 = curfInit->Tcw_ * pw;
                const Eigen::Vector2i px3 =
                    curfInit->cam_->Project2PixelPlane(pc3).cast<int>();
                if (InRange(curfInit->grayImg_, px3)) {
                    //curImg.at<cv::Vec3b>(px2.y(), px2.x()) = {0, 0, 255};
                    // cv::circle(curInitImg, {px3.x(), px3.y()}, 2, color);
                    cv::circle(curInitImg, {px3.x(), px3.y()}, 2,
                               colors.back());
                }

                const Eigen::Vector3d pck = curkf->Tcw_ * pw;
                const Eigen::Vector2i pxk =
                    curkf->cam_->Project2PixelPlane(pck).cast<int>();
                if (InRange(curKFimg, pxk)) {
                    // cv::circle(curKFimg, {pxk.x(), pxk.y()}, 2, color);
                    cv::circle(curKFimg, {pxk.x(), pxk.y()}, 2, colors.back());
                }
            }
        }
    };
    constexpr double pointSize = 1.0;

    vector<Point3d> localPoints;
    vector<cv::Vec3b> localColors;
    localPoints.reserve(interaction->localPoints.size());
    localColors.reserve(localPoints.size());
    cv::Vec3b color1{0, 255, 0};
    GenerateCloud(interaction->localPoints, color1, localPoints, localColors);
    if (!localPoints.empty()) {
        viz::WCloud localCloud(localPoints, localColors);
        if (localColors.empty()) {
            localCloud.setColor({color1});
        }
        localCloud.setRenderingProperty(viz::POINT_SIZE, pointSize);
        window.showWidget("localPointCloud", localCloud);
    }

    vector<Point3d> activePoints;
    vector<cv::Vec3b> activeColors;
    activePoints.reserve(localPoints.size());
    activeColors.reserve(localPoints.size());
    vector<cv::Vec3b> colors1;
    cv::Vec3b color2{0, 0, 255};
    GenerateCloud(interaction->activePoints, color2, activePoints,
                  activeColors);
    if (!activePoints.empty()) {
        viz::WCloud activeCloud(activePoints, activeColors);
        if (activeColors.empty()) {
            activeCloud.setColor({color2});
        }
        activeCloud.setRenderingProperty(viz::POINT_SIZE, pointSize * 2);
        window.showWidget("activePointCloud", activeCloud);
    }

    cout << "local cloud size & active cloud size: " << localPoints.size()
         << " & " << activePoints.size() << endl;

    // 可视化相机pose
    vector<Point3d> startEndCameraPos(2);
    const double coordinateScale =
        0.5 * (vTwc[0].t_wb_ - vTwc[1].t_wb_).head(2).norm();
    for (size_t i = 0; i < vTwc.size(); ++i) {
        // Eigen默认列优先，这里先将其改为行优先以与Mat适配
        Eigen::Matrix<double, 4, 4, Eigen::RowMajor> _Twc =
            vTwc[i].ToMatrix4d();
        double* data = _Twc.data();
        cv::Mat mat44(4, 4, CV_64F, data);
        const cv::Affine3d Twc(mat44);

        if (i == 0 || i == vTwc.size() - 1) {
            const Eigen::Vector3d t = vTwc[i].t_wb_;
            if (i == 0) {
                startEndCameraPos[0] = {t.x(), t.y(), t.z()};
            } else {
                startEndCameraPos[1] = {t.x(), t.y(), t.z()};
            }
        }
        // 显示坐标系
        window.showWidget(fmt::format("cam_{}", i),
                          viz::WCoordinateSystem(coordinateScale), Twc);
    }

    // 创建一个球体表示起点和终点
    if (!config->debugRunOnDesktop) {
        constexpr double radius = 0.001;
        cv::viz::WSphere s0(startEndCameraPos[0], radius, 1, {255, 255, 255});
        cv::viz::WSphere s1(startEndCameraPos[1], radius, 1, {0, 255, 255});
        window.showWidget("S0", s0);
        window.showWidget("S1", s1);
    }

    // 实时显示当前帧投影情况
    constexpr double ratio = 0.5;
    const int w = curImg.cols * ratio, h = curImg.rows * ratio;
    if (curf != nullptr && !config->debugRunOnDesktop) {
        cv::putText(curInitImg, "curInitImage", Point(10, curInitImg.rows - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
        window.showWidget(
            "curInitImage",
            cv::viz::WImageOverlay(curInitImg, cv::Rect(0, 0, w, h)));
        cv::putText(curImg, "curOptImg", Point(10, curInitImg.rows - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
        window.showWidget("curImage", cv::viz::WImageOverlay(
                                          curImg, cv::Rect(w + 10, 0, w, h)));
    }

    if (!config->debugRunOnDesktop) {
        cv::putText(curKFimg, "curKFimg", Point(10, curInitImg.rows - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
        window.showWidget(
            "lastKFimg",
            cv::viz::WImageOverlay(curKFimg, cv::Rect(2 * w + 20, 0, w, h)));
    }

    // 显示轨迹
    vector<cv::Point3d> traj;
    for (const Eigen::Vector3d& p : interaction->trajectory)
        traj.push_back({p.x(), p.y(), p.z()});
    cv::viz::WPolyLine trajPolyline(traj, cv::viz::Color::green());
    window.showWidget("Trajectory", trajPolyline);

    // 显示当前帧的视锥
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K = curf->cam_->K_[0];
    double* dataK = K.data();
    Matx33d intrisicParams(dataK);
    viz::Camera camera(intrisicParams, Size(w, h));
    // 将输入转化为Mat格式
    viz::WCameraPosition camParam;
    Affine3d camPose;
    const double scale = 0.1;
    Eigen::Matrix<double, 4, 4, Eigen::RowMajor> Twc =
        Eigen::Matrix<double, 4, 4>::Identity();
    Twc.block(0, 0, 3, 3) = curf->Twc_.q_wb_.toRotationMatrix();
    Twc.block(0, 3, 3, 1) = curf->Twc_.t_wb_;
    Affine3d matTwc(Twc.data());
    camParam = viz::WCameraPosition(camera.getFov(), curImg, scale,
                                    viz::Color::white());
    camPose = matTwc;
    window.showWidget("cur Camera", camParam, camPose);

    window.registerKeyboardCallback(VizInteraction);
    // 运行事件循环，使窗口响应用户输入
    while (!interaction->resetWindow && curId == interaction->visualCurF.id_ &&
           !interaction->drawEpipolarMatch) {
        window.spinOnce(100);
    }
    interaction->resetWindow = false;
    window.removeAllWidgets();
    //window.close();
}

void ShowCameraCone(const vector<Pose>& vTwc, const vector<Mat>& imgs,
                    const Camera& cam) {
    viz::Viz3d mainWindow("Camera cone window");
    mainWindow.setViewerPose(cv::Affine3d::Identity());

    // 初始化相机类
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K = cam.K_[0];
    double* dataK = K.data();
    Matx33d intrisicParams(dataK);
    const int w = imgs[0].cols, h = imgs[0].rows;
    viz::Camera camera(intrisicParams, Size(w, h));
    cout << "t12: " << vTwc[1].t_wb_.transpose()
         << " norm: " << vTwc[1].t_wb_.norm() << endl;

    // 将输入转化为Mat格式
    vector<viz::WCameraPosition> camParams(imgs.size());
    vector<Affine3d> camPoses(imgs.size());
    const double scale = 0.1;
    for (int i = 0; i < imgs.size(); ++i) {
        Eigen::Matrix<double, 4, 4, Eigen::RowMajor> Twc =
            Eigen::Matrix<double, 4, 4>::Identity();
        Twc.block(0, 0, 3, 3) = vTwc[i].q_wb_.toRotationMatrix();
        Twc.block(0, 3, 3, 1) = vTwc[i].t_wb_;
        Affine3d matTwc(Twc.data());
        if (i == 0)
            camParams[i] = viz::WCameraPosition(camera.getFov(), imgs[i], scale,
                                                viz::Color::white());
        else
            camParams[i] = viz::WCameraPosition(camera.getFov(), imgs[i], scale,
                                                viz::Color::cyan());
        camPoses[i] = matTwc;
        mainWindow.showWidget("Camera" + to_string(i), camParams[i],
                              camPoses[i]);
    }

    const Eigen::Vector3d t1 = vTwc[1].t_wb_;
    const Eigen::Vector2i uv0(320, 210);
    cv::Point3d o0(0, 0, 0), o1(t1[0], t1[1], t1[2]);
    o1 *= 5;
    const Eigen::Vector3d _p = cam.InverseProject(uv0);
    cv::Point3d pc0(_p[0], _p[1], _p[2]);
    cv::viz::WLine l0(o0, pc0, viz::Color::red());
    cv::viz::WLine l1(o0, o1, viz::Color::red());
    cv::viz::WLine l2(pc0, o1, viz::Color::bluberry());
    mainWindow.showWidget("Line0", l0);
    mainWindow.showWidget("Line1", l1);

    // mainWindow.showWidget("Coordinate", viz::WCoordinateSystem(), Affine3d::Identity());
    mainWindow.spin();
}

void InteractionParam::ShowGlobalMapPoint() {
    viz::Viz3d window("Global Point Cloud Viewer");
    cv::Affine3d viewPose;
    window.setViewerPose(viewPose);
    vector<Point3d> points;
    vector<Vec3b> colors;
    if (allMapPoints.empty()) {
        usleep(100 * 1000);
        return;
    }

    for (const Eigen::Vector3d& pw : allMapPoints) {
        points.push_back({pw.x(), pw.y(), pw.z()});

        // {B G R}
        if (pw.z() > 8) {
            colors.push_back({255, 255, 0});
        } else if (pw.z() > 4) {
            colors.push_back({0, 255, 0});
        } else if (pw.z() > 2) {
            colors.push_back({0, 0, 255});
        } else {
            colors.push_back({255, 255, 255});
        }
    }
    cout << endl;
    cout << "show point size: " << points.size() << endl;

    if (points.empty()) {
        cerr << "No Points' depth Converged!" << endl;
        return;
    } else {
        cout << points.size() << " points converged!" << endl;
    }

    // 创建点云对象
    viz::WCloud cloud(points, colors);

    // 设置点云颜色和大小
    // cloud.setColor(cv::viz::Color::green());
    // cloud.setSize(5);

    // 显示点云
    window.showWidget("PointCloud", cloud);

    // 运行事件循环，使窗口响应用户输入
    window.spin();
}
