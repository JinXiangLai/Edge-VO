#include "Utils.h"
#include "Landmark.h"

#include <cstdint>
#include <fstream>
#include <algorithm>

using namespace cv;
using namespace std;

Mat GetDistanceTransform(Mat img) {
    Mat res;
    // https://blog.csdn.net/kakiebu/article/details/82967085
    distanceTransform(img, res, DIST_L2, 0);
    Assert(res.type() == CV_32FC1, "Assert distance transform type error!");
    // TODO: 是否需要归一化呢
    // cv::normalize(res, res, 0, 1);
    return res;
}

vector<Eigen::Vector3d> TransformPoint2Pc(const Pose &T, vector<Eigen::Vector3d> &ps) {
    vector<Eigen::Vector3d> pc;
    for(const Eigen::Vector3d &p : ps) {
        const Eigen::Vector3d t = T * p;
        Assert(t[2] > 0, "[ERROR] depth can't be negative");
        pc.push_back(t);
    }
    return pc;
}

void CaculateDerivative(const Mat &dist, Mat &dx, Mat &dy) {
    const int row = dist.rows;
    const int col = dist.cols;
    dx = Mat(row, col, CV_32FC1, 0.);
    dy = dx.clone();
    for(int i = 0; i < row; ++i) {
        // 遍历一行
        for(int j = 1; j < col-1; ++j) {
             dx.at<float>(i, j) = 0.5 * (dist.at<float>(i, j+1) - dist.at<float>(i, j-1));
            //dx.at<float>(i, j) = (dist.at<float>(i, j+1) - dist.at<float>(i, j));
        }
    }
    for(int j = 0; j < col; ++j) {
        // 遍历一列
        for(int i = 1; i < row-1; ++i) {
             dy.at<float>(i, j) = 0.5 * (dist.at<float>(i+1, j) - dist.at<float>(i-1, j));
            //dy.at<float>(i, j) = (dist.at<float>(i+1, j) - dist.at<float>(i, j));
        }
    }
}

bool InRange(const cv::Mat &img, const Eigen::Vector2i &p) {
    // 边缘行、列忽略
    // return p.x() >= kDescriptorPatchLen && p.x() < img.cols-kDescriptorPatchLen && 
    //         p.y() >= kDescriptorPatchLen && p.y() < img.rows-kDescriptorPatchLen;
    const double imgScale = config->imageScale;
    return p.x() >= 6*imgScale && p.x() < img.cols-6*imgScale && 
            p.y() >= 6*imgScale && p.y() < img.rows-10*imgScale; // 把车头像素滤掉
}

Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d &v) {
    Eigen::Matrix3d m;
    m.setZero();
    m << 0, -v[2], v[1],
         v[2], 0, -v[0],
         -v[1], v[0], 0;
    return m;
}

// TODO: 可以预先对每个像素进行10等分，然后通过查表获得值
double BilinearInterpolate(const cv::Mat &img, const Eigen::Vector2d &p) {
    if(!InRange(img, p.cast<int>())) {
        return 0;
    }
    
    const int col = img.cols;
    const int row = img.rows;
    const int x = int(p.x());
    const int y = int(p.y());
    if(x == col-1 || x == 0 || y == row-1 || y == 0) {
        return img.at<float>(y, x);
    }

    /****** 双线性插值 ******
    * +---+---+
    * + v1+ v2+
    * +---+---+
    * + v3+ v4+
    * +---+---+
    ***********************/
    float v1 = img.at<float>(y, x);
    float v2 = img.at<float>(y, x+1);
    float v3 = img.at<float>(y+1, x);
    float v4 = img.at<float>(y+1, x+1);
    const double wx = p.x() - x;
    const double wy = p.y() - y;
    const double w1 = (1-wx) * (1-wy);
    const double w2 = wx * (1-wy);
    const double w3 = (1-wx) * wy;
    const double w4 = wx * wy;
    // cout << "w1+w2+w3+w4: " << (w1+w2+w3+w4) << endl; // equal to 1
    return w1*v1 + w2*v2 + w3*v3 + w4*v4;
}

void Assert(bool a, const string &s) {
    if(!a) {
        cerr << s << endl;
        exit(-1);
    }
}

void ShowImage(const cv::Mat &img, const string &name, const bool show) {
    if(!show) {
        return;
    }
    Mat m = img.clone();

    m.convertTo(m, CV_32F);
    cv::normalize(m, m, 255, 0, cv::NORM_MINMAX);
    m.convertTo(m, CV_8UC1);
    cv::imshow(name, m);
    cv::waitKey(0);
}

Eigen::Vector3d Quat2RPY(const Eigen::Quaterniond &_q){
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
        pitch = copysign(M_PI / 2, sinp); // 使用90度或-90度  
    else  
        pitch = asin(sinp);  
  
    // yaw (z-axis rotation)  
    double siny_cosp = 2.0 * (w * z + x * y);  
    double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);  
    const double yaw = atan2(siny_cosp, cosy_cosp);
    return {roll, pitch, yaw};
}

ostream& operator<<(ostream &cout, const Pose& T){
    cout << setprecision(2) << "RPY | t: " << Quat2RPY(T.q_wb_).transpose() * kRad2Deg 
         << " | " << T.t_wb_.transpose();
    return cout;
}

Mat DrawMatch(const Mat &img1, const Mat &img2, const vector<Eigen::Vector2d> &kp1, const vector<Eigen::Vector2d> &kp2,
                const string &name, const int ratio, const int jump) {
    if(kp1.empty() || kp2.empty()) {
        cerr << "{"+name << "} Error!" << endl
             << "kp1 & kp2 size: " << kp1.size() << " & " << kp2.size() << endl;
        exit(-1);
    }
    Assert(kp1.size()==kp2.size() || kp1.size() == 1 || kp1.size() < kp2.size(), "match point size error!");
    Mat im(max(img1.rows, img2.rows), (img1.cols+img2.cols), CV_8UC3, cv::Scalar{0, 0, 0});
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

    for(int i = 0; i < kp2.size(); ++i) {
        if(!InRange(img2, kp2[i].cast<int>())) {
            continue;
        }
        
        cv::Point c1(kp1[0].x()*ratio, kp1[0].y()*ratio);
        if(kp1.size() == kp2.size()) {
            c1 = cv::Point (kp1[i].x()*ratio, kp1[i].y()*ratio);
        }

        cv::Point c2(sCol+kp2[i].x()*ratio, kp2[i].y()*ratio);
        // if((kp1.size() == 1 || 1) && i%10 == 0) {
        if( i%jump == 0 ) {
            cv::line(im, c1, c2, lColor, 1);
        }
        cv::circle(im, c1, radius, pColor, 1);
        cv::circle(im, c2, radius, pColor, 1);
    }
    cv::namedWindow(name);
    cv::imshow(name, im);
    cv::imwrite(name+".png", im);
    cv::waitKey(0);
    return im;
}

vector<Eigen::Vector2d> FindMatches(const Landmark &pc1, const KeyFrame &kf2, const Pose &T21, const Camera &cam) {
    const Eigen::Vector2d &kp1 = pc1.uv_;
    const Mat &edgeImg = kf2.edgeImg_[0];
    const u_int64_t d1 = pc1.descriptor_;

    /******** 使用极线约束寻找匹配关键点 ********
    * R21 * s1 * Pc1_norm + t21 = s2 * Pc2_norm
    * R21 * s1/s2 * Pc1_norm + 1/s2 * t21 = Pc2_norm
    * [t21]x * R21 * s1/s2 * Pc1_norm = [t21]x * Pc2_norm
    * Pc2_norm.T * [t21]x * R21 * s1/s2 * Pc1_norm = 0
    * Pc2_norm.T * [t21]x * R21 * Pc1_norm = 0 --------> 归一化平面上极线约束
    * [K.inv * px2].T * [t21]x * R21 * Pc1_norm = 0 ---> 像素平面上的极线约束 
    ********************************************/
    Eigen::Vector3d _c = skewSymmetric(T21.t_wb_) * T21.q_wb_.toRotationMatrix() * cam.InverseProject(kp1.cast<int>());
    // |k00 k01 k02|   |x|   |k00*x + k01*y + k02|
    // |k10 k11 k12| * |y| = |k10*x + k11*y + k12|
    // |k20 k21 k22|   |1|   |k20*x + k21*y + k22|
    //                                                                   |c0|
    // [k00*x + k01*y + k02, k10*x + k11*y + k12, k20*x + k21*y + k22] * |c1| = 
    //                                                                   |c2|
    // k00*c0*x + k01*c0*y + k02*c0 +
    // k10*c1*x + k11*c1*y + k12*c1 +
    // k20*c2*x + k21*c2*y + k22*c2 = (k00*c0 + k10*c1 + k20*c2)*x +
    //                                (k01*c0 + k11*c1 + k21*c2)*y +
    //                                (k02*c0 + k12*c1 + k22*c2)
    // 像素平面上极线约束的参数
    const Eigen::Matrix3d Ki = cam.K_inv_;
    const double k00 = Ki.row(0)[0], k01 = Ki.row(0)[1], k02 = Ki.row(0)[2],
                 k10 = Ki.row(1)[0], k11 = Ki.row(1)[1], k12 = Ki.row(1)[2],
                 k20 = Ki.row(2)[0], k21 = Ki.row(2)[1], k22 = Ki.row(2)[2];
    const double c0 = k00*_c[0] + k10*_c[1] + k20*_c[2],
                 c1 = k01*_c[0] + k11*_c[1] + k21*_c[2],
                 c2 = k02*_c[0] + k12*_c[1] + k22*_c[2];
    // c[0]*x + c[1]*y + c[2] = 0
    // y = -c[0]/c[1]*x - c[2]/c[1]
    
    auto Kp2Useful = [&kf2, &d1](const Eigen::Vector2i &p2, int &score) -> bool {
        const Mat &edgeImg = kf2.edgeImg_[0];
        const int x=p2[0], y=p2[1];

        if(!InRange(edgeImg, p2) || (edgeImg.at<uchar>(y, x) != 0)) {
            return false;
        }

        const int descId = kf2.pointMapId_.at({x, y});
        const u_int64_t d2 = kf2.descriptor_[descId];
        score = CalculateDescriptorScore(d1, d2);
        return score < config->maxDescriptorDist;

        // return (edgeImg.at<uchar>(y, x) == 0);
        // return InRange(edgeImg, p) && (edgeImg.at<uchar>(y-1, x) != 255
        //     || edgeImg.at<uchar>(y, x) != 255 || edgeImg.at<uchar>(y+1, x) != 255
        //     || edgeImg.at<uchar>(y, x-1) != 255 || edgeImg.at<uchar>(y, x+1) != 255);
    };

    Eigen::Vector2i xRange, yRange;
    GetProjectRange(pc1, T21, cam, xRange, yRange);
    const int minCol = max(xRange[0], 0);
    const int maxCol = min(xRange[1], edgeImg.cols);
    const int minRow = max(yRange[0], 0);
    const int maxRow = min(yRange[1], edgeImg.rows);

    vector<Eigen::Vector2d> kp2;
    vector<pair<int, Eigen::Vector2i> > scoreKp2;
    auto IsZero = [](const double a) -> bool {return abs(a) < 1e-10;};
    const double roundOff = 0.5; // 0.5 四舍五入参数

    if(IsZero(c1) && IsZero(c0)) {
        return kp2; 
    } else if(IsZero(c1) && !IsZero(c0) ) {
        const int x = -c2/c0 + roundOff;
        cout << "epilor line col: " << x << endl;
        for(int y = minRow; y < maxRow; ++y) {
            const Eigen::Vector2i px{x, y};
            int score = INT_MAX;
            if(Kp2Useful(px, score) ) {
                scoreKp2.push_back(make_pair(score, px));
            }
        }
    } else if(!IsZero(c1) && IsZero(c0) ) {
        const int y = -c2/c1 + roundOff;
        cout << "epilor line row: " << y << endl;
        for(int x = minCol; x < maxCol; ++x) {
            const Eigen::Vector2i px{x, y};
            int score = INT_MAX;
            if(Kp2Useful(px, score) ) {
                scoreKp2.push_back(make_pair(score, px));
            }
        }
    } else {
        // y = ax + b
        const double a = -c0/c1, b = -c2/c1;
        for(int x = minCol; x < maxCol; ++x) {
            const int y = a*x + b + roundOff;
            const Eigen::Vector2i px{x, y};
            int score = INT_MAX;
            if(Kp2Useful(px, score) ) {
                scoreKp2.push_back(make_pair(score, px));
            }
        }
    }

    sort(scoreKp2.begin(), scoreKp2.end(), [](const pair<int, Eigen::Vector2i> &p1, 
            const pair<int, Eigen::Vector2i> &p2) -> bool {return p1.first < p2.first;} );
    for(int i = 0; i < 3 && i < scoreKp2.size(); ++i) {
        kp2.emplace_back(scoreKp2[i].second.cast<double>());
    }
    return kp2;
}

// 比像素平面上的极线约束多了N次投影到像素平面的运算
vector<Eigen::Vector2d> FindMatchesWithEpipolarConstraintOnImagePlane(const Eigen::Vector2d &kp1, const Mat &edgeImg, 
    const Pose &T21, const Camera &cam) {
    /******** 使用极线约束寻找匹配关键点 ********
    * R21 * s1 * Pc1_norm + t21 = s2 * Pc2_norm
    * R21 * s1/s2 * Pc1_norm + 1/s2 * t21 = Pc2_norm
    * [t21]x * R21 * s1/s2 * Pc1_norm = [t21]x * Pc2_norm
    * Pc2_norm.T * [t21]x * R21 * s1/s2 * Pc1_norm = 0
    * Pc2_norm.T * [t21]x * R21 * Pc1_norm = 0 --------> 归一化平面上极线约束
    * fx*x+cx = px_x ==> 归一化平面上[1/fx]米对应1个像素
    * fy*y+cy = px_y 
    ********************************************/
    Eigen::Vector3d c = skewSymmetric(T21.t_wb_) * T21.q_wb_.toRotationMatrix() * cam.InverseProject(kp1.cast<int>());
    // c[0]*x + c[1]*y + c[2] = 0 ==> 归一化平面上的极线
    // y = -c[0]/c[1]*x - c[2]/c[1]
    
    auto Kp2Useful = [&edgeImg](const Eigen::Vector2i &p) -> bool {
        const int x=p[0], y=p[1];
        // InRange函数保证(y, x)点的十字架处像素不会超过索引
        return InRange(edgeImg, p) && (!edgeImg.at<uchar>(y, x)
                || !edgeImg.at<uchar>(y-1, x) || !edgeImg.at<uchar>(y+1, x)
                || !edgeImg.at<uchar>(y, x-1) || !edgeImg.at<uchar>(y, x+1));
    };

    vector<Eigen::Vector2d> kp2;
    auto IsZero = [](const double a) -> bool {return abs(a) < 1e-10;};

    const double xStep = 1.0/cam.fx_, yStep = 1.0/cam.fy_;
    const double xStart = -cam.cx_/cam.fx_ + xStep, yStart = -cam.cy_/cam.fy_ + yStep;
    const double xEnd = -xStart, yEnd = -yStart;

    if(IsZero(c[1]) && IsZero(c[0])) {
        return kp2; 
    } else if(IsZero(c[1]) && !IsZero(c[0]) ) {
        const double x = -c[2]/c[0]; // 归一化平面坐标
        cout << "x: " << x << endl;
        for(double y = yStart; y < yEnd; y+=yStep) {
            const Eigen::Vector2d px = cam.Project2PixelPlane({x, y, 1}) ;
            if(Kp2Useful(px.cast<int>()) ) {
                kp2.push_back(px.cast<double>() );
            }
        }
    } else if(!IsZero(c[1]) && IsZero(c[0]) ) {
        const double y = -c[2]/c[1];
        for(double x = xStart; x < xEnd; x+=xStep) {
            const Eigen::Vector2d px = cam.Project2PixelPlane({x, y, 1});
            if(Kp2Useful(px.cast<int>()) ) {
                kp2.push_back(px.cast<double>() );
            }
        }
    } else {
        // y = ax + b
        const double a = -c[0]/c[1], b = -c[2]/c[1];
        for(double x = xStart; x < xEnd; x+=xStep) {
            const double y = a*x + b;
            const Eigen::Vector2d px = cam.Project2PixelPlane({x, y, 1});
            if(Kp2Useful(px.cast<int>()) ) {
                kp2.push_back(px.cast<double>() );
            }
        }
    }
    return kp2;
}


Eigen::Vector3d Triangulate(const Eigen::Vector2d &kp2, const Pose &T21, const Camera &cam) {
    /******** 三角化地图点 ********
    * R21 * Pc1 + t21 = z2 * kp2_norm
    * [kp2_norm]x * R21 * Pc1 = -[kp2_norm]x * t21
    *****************************/
    // TODO: 检验为什么该种三角化方式不行！！！
    // 直观理解就是，与kp2出发射线有交点的地图点均可满足该约束，因为没有用到kp1信息，故而无法确定Pc1
    const Eigen::Vector3d kp2Norm = cam.InverseProject(kp2.cast<int>());
    const Eigen::Matrix3d skew = skewSymmetric(kp2Norm);
    const Eigen::Matrix3d A = skew * T21.q_wb_.toRotationMatrix();
    const Eigen::Vector3d b = -skew * T21.t_wb_;
    return A.colPivHouseholderQr().solve(b);
    //return A.inverse() * b;
}

Eigen::Vector3d Triangulate(const Eigen::Vector2d &kp1, const Eigen::Vector2d &kp2, const Pose &T21, const Camera &cam) {
    /******** 三角化地图点 Pc1 ********
    * R1w * Pw + t1w = z * kp1_norm
    * [kp1_norm]x * R1w * Pw = -[kp1_norm]x * t1w
    *
    * R21 * Pc1 + t21 = z * kp2_norm
    *****************************/
    const Eigen::Vector3d kp1Norm = cam.InverseProject(kp1.cast<int>());
    const Eigen::Vector3d kp2Norm = cam.InverseProject(kp2.cast<int>());
    const Eigen::Matrix3d skew1 = skewSymmetric(kp1Norm);
    const Eigen::Matrix3d skew2 = skewSymmetric(kp2Norm);
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

bool UpdateLandmarkDepth(const vector<Eigen::Vector2d> &kp2, const Pose &T21, const Camera &cam, Landmark &landmark) {
    // TODO:需要根据现实条件实现该函数，如使用光度残差作为阈值
    const Pose T12 = T21.Inverse();
    auto SolutionInrange = [&landmark, &T12](const double pcZ) -> bool {
                                // 进行深度滤波
                                const Eigen::Vector3d v1 = landmark.cam_->InverseProject(landmark.uv_.cast<int>(), 
                                   pcZ);
                                const Eigen::Vector3d v2 = v1 - T12.t_wb_;
                                const double ang = acos(v1.dot(v2)/v1.norm()/v2.norm()) * kRad2Deg;
                                return pcZ >= landmark.depthRange_[0] && 
                                       pcZ <= landmark.depthRange_[1] &&
                                       pcZ > config->minDepth && 
                                       pcZ < config->maxDepth &&
                                       ang > config->minGoodTriangulateAngle;
                            };

    const Eigen::Vector2d kp1 = landmark.uv_;
    vector<double> depth;
    // cout << "current triangulated depth: ";
    double sumDepth = 0, maxDepth = 0, minDepth = DBL_MAX;
    Eigen::Vector2d specialPc2;
    for(const Eigen::Vector2d &p : kp2) {

       const Eigen::Vector3d pc1 = Triangulate(kp1, p, T21, cam);
    //    cout << pc1.z() << " ";
        if(SolutionInrange(pc1.z()) ) {
            depth.push_back(pc1.z());
            if(pc1.z() < minDepth) {
                minDepth = pc1.z();
            }
            if(pc1.z() > maxDepth) {
                maxDepth = pc1.z();
            }
            sumDepth += pc1.z();
            specialPc2 = p;
        }
    }
    // cout << endl;

    double u2 = -1, cov2 = -1;
    if(depth.size() > 1) {
        // TODO: 这里应该如何更新呢？
        u2 = sumDepth / depth.size();
        cov2 = pow(0.5 * (maxDepth - minDepth), 2);
    } else if(depth.size() == 1 ) {
        const double std = GetOnePixelUncertainty(T21.Inverse().t_wb_, 
            cam.InverseProject(kp1.cast<int>(), depth[0]), cam.fx_);
        u2 = depth[0], cov2 = pow(std, 2);
    } else {
        return false;
    }

    const double u1 = landmark.z_, cov1 = landmark.depthCov_;
    // cout << "maxDepth, minDepth, depth size: " << maxDepth << " " << minDepth << " " << depth.size() << endl;
    // 信息融合，标准差一直减小
    landmark.z_ = (u2*cov1 + u1*cov2) / (cov1 + cov2);
    landmark.depthCov_ = (cov1 * cov2)/(cov1 + cov2);
    landmark.UpdateUncertainty();
    // cout << "u1, u2, cov1, cov2, z: " << u1 << " " << u2 << " " << cov1 << " " << cov2 
    //     << " " << landmark.z_ << endl;
    
    static ofstream unf;
    static int num = 0;
    if(!num) {
        unf.open("depth_uncertainty.csv");
        unf.close();
    }
    unf.open("depth_uncertainty.csv", ios::app);
    unf << fixed << &landmark << " [" << landmark.depthRange_[0] << ", " << landmark.depthRange_[1] << "] std, depth: " 
        << landmark.uncertainty_ << " " << landmark.z_ << endl;
    unf.close();
    return true;
}

Pose ConvertRPYandPostion2Pose(const Eigen::Vector3d &rpy, const Eigen::Vector3d &t, const double deg2rad) {
    Eigen::Quaterniond q_c1c2 = Eigen::AngleAxisd(rpy[2] * deg2rad, Eigen::Vector3d::UnitY())
                              * Eigen::AngleAxisd(rpy[1] * deg2rad, Eigen::Vector3d::UnitX())
                              * Eigen::AngleAxisd(rpy[0] * deg2rad, Eigen::Vector3d::UnitZ());
    return Pose(q_c1c2, t);
}

void varifyTriangulate() {
    Pose Twc1(Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());
    Pose Tc1c2 = ConvertRPYandPostion2Pose({0, 0, 20}, {0.5, 1., -0.5}, kDeg2Rad);
    shared_ptr<Camera> cam = make_shared<Camera>(config);
    
    const Eigen::Vector2d px1{15, 17};
    const double z1 = 5.0;
    const Eigen::Vector3d pc1 = cam->InverseProject(px1.cast<int>(), z1);
    cout << "pc1: " << pc1.transpose() << endl;
    cout << "px1: " << cam->Project2PixelPlane(pc1).transpose() << endl;

    const Eigen::Vector3d pc2 = Tc1c2.Inverse() * pc1;
    cout << "pc2: " << pc2.transpose() << endl;
    const Eigen::Vector2d px2 = cam->Project2PixelPlane(pc2);
    cout << "px2: " << px2.transpose() << endl; 

    // 验证像素偏差对三角化精度的影响
    vector<Eigen::Vector2d> px2_9;
    for(int x = -1; x < 2; ++x) {
        for(int y = -1; y < 2; ++y) {
            Eigen::Vector2d px{px2.x()+x, px2.y()+y};
            px2_9.push_back(px);
        }
    }

    for(int i = 0; i < 9; ++i) {
        Eigen::Vector3d pc1_est = Triangulate(px1, px2_9[i], Tc1c2.Inverse(), *cam);
        cout << "pc1_est: " << pc1_est.transpose() << endl;
    }
    // 结论：在[3x3]邻域范围内，对三角化精度的影响尚可接受
}

size_t LoadImages(const string& strDirectory, vector<string>& vstrImages, vector<double>& vTimeStamps) {
    const string imageDirectory = strDirectory + "/image";
    const string imgSuffix = ".jpg";
    const string imageTimestampFile = strDirectory + "/image_timestamp.csv";
    
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

size_t LoadPriorOdom(const string &strDirectory, vector<Eigen::Matrix<double, 8, 1>> &vPriorPose) {
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
        if (input.empty()) {
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

void FindImageAndPose(const int idx, const vector<string> & vstrImages, const vector<double> vTimeStamps, 
    const vector<Eigen::Matrix<double, 8, 1>> vPriorPose, const WheelCameraCalib &calib, vector<Mat> &imgs, 
    vector<Pose> &vTwc, const int needNum) {
    Assert(!vstrImages.empty() && !vTimeStamps.empty() && !vPriorPose.empty(), "Dataset is empty!!!");

    // 插值pose
    auto InterpolatePose = [&vPriorPose, &calib] (const double &t) -> Pose {
        if(t < vPriorPose.front()[0] || t > vPriorPose.back()[0]) {
            cerr << "Can't find match pose!!!" << endl;
            exit(-1);
        }
        Eigen::Matrix<double, 8, 1> Twv1, Twv2;
        for(int i = 1; i < vPriorPose.size(); ++i) {
            if(vPriorPose[i][0] > t) {
                Twv1 = vPriorPose[i-1];
                Twv2 = vPriorPose[i];
                break;
            }
        }
        const double ratio = (t-Twv1[0])/(Twv2[0]-Twv1[0]);
        const Eigen::Vector3d p = (1-ratio)*Twv1.middleRows(1, 3) + ratio*Twv2.middleRows(1, 3);
        const Eigen::Quaterniond q1(Twv1[7], Twv1[4], Twv1[5], Twv1[6]);
        const Eigen::Quaterniond q2(Twv2[7], Twv2[4], Twv2[5], Twv2[6]);
        const Eigen::Quaterniond q = q1.slerp(ratio, q2);
        Pose Twv(q, p);
        return calib.Tcv_ * Twv * calib.Tvc_;
    };

    imgs.push_back(cv::imread(vstrImages[idx], IMREAD_GRAYSCALE));
    const double imgScale = config->imageScale;
    const int newW = imgs[0].cols * imgScale, newH = imgs[0].rows * imgScale;
    cv::resize(imgs[0], imgs[0], cv::Size(newW, newH) );
    vTwc.push_back(InterpolatePose(vTimeStamps[idx] + config->imgTimeOffset) );

    int id = idx;
    int curId = 1;
    
    while (id < vstrImages.size() && curId < needNum) {
        // 避免死循环
        ++id;

        const double time = vTimeStamps[id];
        Pose Twc_i = InterpolatePose(time);
        if((Twc_i.Inverse() * vTwc[curId-1]).t_wb_.norm() > config->minTranslation) {
            imgs.push_back(cv::imread(vstrImages[id], IMREAD_GRAYSCALE));
            cv::resize(imgs[curId], imgs[curId], cv::Size(newW, newH) );
            vTwc.push_back(Twc_i);
            ++curId;
        }
    }
    Assert(imgs.size() == needNum && vTwc.size() == needNum, "Find imgs and Twc size Error !!!");
}


void GetImageAndPose(const int idx, const vector<string> &vstrImages, const vector<double> vTimeStamps, 
    const vector<Eigen::Matrix<double, 8, 1>> vPriorPose, const WheelCameraCalib &calib, cv::Mat &img, Pose &Twc){
    Assert(!vstrImages.empty() && !vTimeStamps.empty() && !vPriorPose.empty(), "Dataset is empty!!!");

    // 插值pose
    auto InterpolatePose = [&vPriorPose, &calib] (const double &t) -> Pose {
        if(t < vPriorPose.front()[0] || t > vPriorPose.back()[0]) {
            cerr << "Can't find match pose!!!" << endl;
            exit(-1);
        }
        Eigen::Matrix<double, 8, 1> Twv1, Twv2;
        for(int i = 1; i < vPriorPose.size(); ++i) {
            if(vPriorPose[i][0] > t) {
                Twv1 = vPriorPose[i-1];
                Twv2 = vPriorPose[i];
                break;
            }
        }
        const double ratio = (t-Twv1[0])/(Twv2[0]-Twv1[0]);
        const Eigen::Vector3d p = (1-ratio)*Twv1.middleRows(1, 3) + ratio*Twv2.middleRows(1, 3);
        const Eigen::Quaterniond q1(Twv1[7], Twv1[4], Twv1[5], Twv1[6]);
        const Eigen::Quaterniond q2(Twv2[7], Twv2[4], Twv2[5], Twv2[6]);
        const Eigen::Quaterniond q = q1.slerp(ratio, q2);
        Pose Twv(q, p);
        return calib.Tcv_ * Twv * calib.Tvc_;
    };

    img = cv::imread(vstrImages[idx], IMREAD_GRAYSCALE);
    const double imgScale = config->imageScale;
    const int newW = img.cols * imgScale, newH = img.rows * imgScale;
    cv::resize(img, img, cv::Size(newW, newH) );
    Twc = InterpolatePose(vTimeStamps[idx] + config->imgTimeOffset);
}

//double CalculateScore(const Eigen::Matrix<float, kDescriptorPatchSize, 1> &d1, const Eigen::Matrix<float, kDescriptorPatchSize, 1> &d2) {
//    /****************
//    * +---+---+---+
//    * + 1 + 2 + 1 +
//    * +---+---+---+
//    * + 2 + 3 + 2 +
//    * +---+---+---+
//    * + 1 + 2 + 1 +
//    * +---+---+---+
//    *****************/
//    constexpr double ratio = 1.0/15;
//    const Eigen::Matrix<float, kDescriptorPatchSize, 1> d = (d1-d2).cwiseAbs();
//    const double cost = d[0] + 2*d[1] + d[2] +
//                        2*d[3] + 3*d[4] + 2*d[5] +
//                        d[6] + 2*d[7] + d[8];
//    // return (cost/15)/9;
//    return cost * ratio;
//}

uint64_t CalculateDescriptor(const Mat &grayImg, const Eigen::Vector2i &px) {
    // 返回descDim维描述子, [8x8]的范围内对角线位置的像素值比值
    uint64_t des = 0;
    const int ox = px.x(),   oy = px.y(),
                sx = -4, ex = 4,
                sy = -4, ey = 3;
    int move = -1;
    for(int i = sx; i <= ex; ++i) {
        for(int j = sy; j <= ey; ++j) {
            const uchar v1 = grayImg.at<uchar>(oy+j, ox+i);
            const uchar v2 = grayImg.at<uchar>(oy-j, ox-i);
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

int CalculateDescriptorScore(const uint64_t v1, const uint64_t v2) {
    // 异或，相同值0，不同值为1,意味着score越小越匹配
    uint64_t d = v1^v2;
    int sum = 0;
    for(int i = 0; i < KeyFrame::descDim; ++i) {
        sum += bool((d>>i)&1);
    }
    // cout << "sum: " << sum << endl;
    return sum;
}

void GetProjectRange(const Landmark &lp, const Pose& T21, const Camera &cam, Eigen::Vector2i &xRange, 
    Eigen::Vector2i &yRange) {
    const Eigen::Vector3d pc1_1 = cam.InverseProject(lp.uv_.cast<int>(), lp.depthRange_[0]);
    const Eigen::Vector3d pc1_2 = cam.InverseProject(lp.uv_.cast<int>(), lp.depthRange_[1]);
    const Eigen::Vector3d pc2_1 = T21 * pc1_1;
    const Eigen::Vector3d pc2_2 = T21 * pc1_2;
    const Eigen::Vector2i uv1 = cam.Project2PixelPlane(pc2_1).cast<int>();
    const Eigen::Vector2i uv2 = cam.Project2PixelPlane(pc2_2).cast<int>();
    if(uv1[0] > uv2[0]) {
        xRange[0] = uv2[0];
        xRange[1] = uv1[0];
    } else {
        xRange[0] = uv1[0];
        xRange[1] = uv2[0];
    }

    if(uv1[1] > uv2[1]) {
        yRange[0] = uv2[1];
        yRange[1] = uv1[1];
    } else {
        yRange[0] = uv1[1];
        yRange[1] = uv2[1];
    }
}

double GetOnePixelUncertainty(const Eigen::Vector3d &t12, const Eigen::Vector3d &pc1, const double f) {
    const Eigen::Vector3d pc2 = pc1 - t12;
    const double pc1Norm = pc1.norm(), t12Norm = t12.norm(), 
        d1 = pc1Norm * t12Norm, d2 = pc2.norm() * t12Norm;
    const double alpha = acos(pc1.dot(t12)/d1);
    const double belta = acos(pc2.dot(-t12)/d2);
    const double deltaBelta = atan2(config->filterPixelError, f);

    const double belta2 = belta + deltaBelta;
    const double gamma = M_PI - alpha - belta2;
    const double newDepth = t12Norm * sin(belta2) / sin(gamma);

    return abs(pc1Norm - newDepth);
}

bool NeedNewKF(const KeyFrame *kf, const KeyFrame *f) {
    const Pose T12 = kf->priorTwc_.Inverse() * f->Twc_;
    return T12.t_wb_.norm() > config->needNewKFtrans 
        || Quat2RPY(T12.q_wb_).norm() * kRad2Deg > config->needNewKFrot;
}

bool IsFastPoint(const cv::Mat &gray, const Eigen::Vector2i px) {
    const Point2i pt{px.x(), px.y()};
    const int v = gray.at<uchar>(pt);

    int maxNum = 0;
    int minNum = 0;
    for(int i = 0; i < 16; ++i) {
        const Point2i pt2 {pt.x+FASTpoint[i][0], pt.y+FASTpoint[i][1]};
        const int v2 = gray.at<uchar>(pt2);
        const int diff = v - v2;
        if(diff > config->fastTh) {
            ++maxNum;
        } else if(diff < config->fastTh) {
            ++minNum;
        }
    }
    return maxNum > 11 || minNum > 11;
}

int DrawMatch(KeyFrame *kf1, KeyFrame *kf2, const std::string &name) {
    if(kf1 == kf2) {
        return 0;
    }
    std::vector<Eigen::Vector2d> px1, px2;

    for(Landmark *p : kf1->landmark_) {
        if(p->target_.count(kf2)) { 
            // 说明还是将host也加入相互观测方便
            px1.push_back(p->target_[kf1]);
            px2.push_back(p->target_[kf2]);
        }
    }
    if(px1.empty()) {
        px1.push_back({0, 0});
        px2.push_back({0, 0});
    }
    DrawMatch(kf1->edgeImg_[0], kf2->edgeImg_[0], px1, px2, name);
    return px1.size();
}

int DrawMatch(vector<Landmark*> &ps, KeyFrame *kf2, const std::string &name) {
    vector<Eigen::Vector2d> Px1, Px2;
    shared_ptr<Camera> cam = kf2->cam_;
    for(int i = 0; i < ps.size(); ++i) {
        Landmark *p = ps[i];
        KeyFrame *host = p->host_;
        const Eigen::Vector3d pc1 = p->GetPc();
        const Eigen::Vector3d pw = host->Twc_ * pc1;

        const Eigen::Vector3d pc2 = kf2->Tcw_ * pw;
        if(pc2.z() < config->minDepth || pc2.z() > config->maxDepth) {
            continue;
        }
        const Eigen::Vector2d px2 = cam->Project2PixelPlane(pc2);
        if(!InRange(kf2->dist_[0], px2.cast<int>()) ) {
            continue;
        }

        Px1.push_back(p->uv_);
        Px2.push_back(px2);
    }
    DrawMatch(ps[0]->host_->edgeImg_[0], kf2->edgeImg_[0], Px1, Px2, name);
    return Px1.size();
}


void ShowPointCloud(const vector<Landmark*> &ps) {
    viz::Viz3d window("Point Cloud Viewer");
    cv::Affine3d viewPose;
    window.setViewerPose(viewPose);
    vector<Point3d> points;
    vector<Vec3b> colors;

    for(Landmark *_p : ps) {
        const Landmark &p = *_p;
        if(!p.Converge()) {
            continue;
        }
        // const double depth = p.depthRange_[0]; // p.z_;//0.5 * (p.depthRange_[0] + p.depthRange_[1]);
        // const Eigen::Vector3d &pc = p.cam_->InverseProject(p.uv_.cast<int>(), depth);
        const Eigen::Vector3d &pc = p.GetPw();
        points.push_back({pc.x(), pc.y(), pc.z()});
        // cout << "[" << p.depthRange_[0] << " " << p.depthRange_[1] << "]  ";

        // {B G R}
        if(pc.z() > 8) {
            colors.push_back({255, 255, 0});
        } else if(pc.z() > 4) {
            colors.push_back({0, 255, 0});
        } else if(pc.z() > 2) {
            colors.push_back({0, 0, 255});
        } else {
            colors.push_back({255, 255, 255});
        }
        //cout << pc.z() << ", ";
    }
    cout << endl;
    cout << "show point size: " << points.size() << endl;

    if(points.empty() ) {
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

void ShowPointCloud(const vector<Landmark* > &ps1, const vector<Landmark* > &ps2, 
    const std::string &windowName, const double zOffset) {
    viz::Viz3d window(windowName);
    cv::Affine3d viewPose;
    window.setViewerPose(viewPose);
    vector<Point3d> points1, points2;

    auto Generate = [](const vector<Landmark*> &ps, vector<Point3d> &points) {
        for(const Landmark* p : ps) {
            if(p==nullptr || !p->Converge()) {
                continue;
            }
            const Eigen::Vector3d &pw = p->GetPw();
            points.push_back({pw.x(), pw.y(), pw.z()});
        }

        if(points.empty()) {
            cerr << "No Points' depth Converged!" << endl;
            return;
        } else {
            cout << points.size() << " points converged!" << endl;
        }
        cout << "show points size: " << points.size() << endl;
    };

    Generate(ps1, points1);
    Generate(ps2, points2);
    vector<Vec3b> colors1(points1.size(), {0, 0, 255}), colors2(points2.size(), {0, 255, 0});
    for(auto &p : points1) {
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

void ShowPointCloud(const set<Landmark* > &ps) {
    if(ps.empty()) {
        return;
    }
    viz::Viz3d window("LocalMap Viewer");
    cv::Affine3d viewPose;
    window.setViewerPose(viewPose);
    vector<Point3d> points;

    for(Landmark *p : ps) {
        if(p == nullptr || !p->Converge()) {
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