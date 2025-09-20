#include <iostream>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

constexpr double kDeg2Rad = 180 * M_1_PI;

/******** 问题1：ORB特征的旋转不变性是如何计算的 ********
* step1: 生成一张图像；
* step2: 通过旋转矩阵R对图像进行仿射变换
* step3: 使用R‘对图像进行逆仿射变换，看得到的像素坐标前后变化差异
* 注意：其实问题只有一个：就是由于图像是(int, int)型坐标系，所以可能存在一些精度损失而已，
*      在相机采集图像中，由于像素值是渐变的 结果仍会是带误差但趋于正确的
***************************************************/

/******** 问题2：已知水平姿态角，如何计算出具有旋转不变性的ORB特征 ********
* step1: 利用roll、pitch解算出图像绕其Z轴的旋转角对应的旋转矩阵R
* step2: 利用R’旋转图像，并计算最终旋转后的描述子
*****************************************************************/

int main(int argc, char** argv) {
    if (argc != 2) {
        cerr << "Usage: ./orb ang(deg)" << endl;
        exit(-1);
    }
    const double ang = stof(argv[1]) * kDeg2Rad;
    const double ca = cos(ang), sa = sin(ang);
    //double data[4] = {ca, -sa, sa, ca};
    //Mat R(2, 2, CV_32F, data);
    //Mat invR = R.inv();

    const int imgCol = 10, imgRow = 10;
    const int halfCol = imgCol / 2, halfRow = imgRow / 2;
    Mat img(imgRow, imgCol, CV_32F);
    for (int i = 0; i < imgRow; ++i) {
        for (int j = 0; j < imgCol; ++j) {
            img.at<float>(i, j) = i * imgRow + j / 2;
        }
    }
    cout << setprecision(1) << "img:\n" << img << endl;

    Mat img2(imgRow, imgCol, CV_32F, 0.);
    for (int x = 0; x < imgCol; ++x) {
        for (int y = 0; y < imgRow; ++y) {
            Point2i p(x - halfCol, y - halfRow);
            Point2i p2(ca * p.x - sa * p.y + halfCol,
                       sa * p.x + ca * p.y + halfRow);
            p.x += halfCol;
            p.y += halfRow;
            // Point2i 类型索引为{x, y}
            // 旋转后的图像 = 旋转前图像对应坐标的像素
            //cout << "p1: {" << p.x << ", " << p.y << "}" << endl;
            //cout << "p2: {" << p2.x << ", " << p2.y << "}" << endl;
            if (p2.x >= 0 && p2.x < imgCol && p2.y >= 0 && p2.y < imgRow) {
                // 由于整数，所以肯定很多像素无法得到填充
                img2.at<float>(p2) = img.at<float>(p);
            }
        }
    }
    cout << fixed << setprecision(1) << "img2:\n" << img2 << endl;

    Mat img3(imgRow, imgCol, CV_32F, 0.);
    for (int x = 0; x < imgCol; ++x) {
        for (int y = 0; y < imgRow; ++y) {
            Point2i p(x - halfCol, y - halfRow);
            Point2i p2(ca * p.x + sa * p.y + halfCol,
                       -sa * p.x + ca * p.y + halfRow);
            p.x += halfCol;
            p.y += halfRow;
            // Point2i 类型索引为{x, y}
            // 旋转后的图像 = 旋转前图像对应坐标的像素
            //cout << "p1: {" << p.x << ", " << p.y << "}" << endl;
            //cout << "p2: {" << p2.x << ", " << p2.y << "}" << endl;
            if (p2.x >= 0 && p2.x < imgCol && p2.y >= 0 && p2.y < imgRow) {
                // 由于整数，所以肯定很多像素无法得到填充
                img3.at<float>(p2) = img2.at<float>(p);
            }
        }
    }
    cout << fixed << setprecision(1) << "img3:\n" << img3 << endl;

    Eigen::Matrix3d m;
    m << 1, 2, 3, 4, 5, 6, 7, 8, 9;
    cout << "m.hasNaN(): " << m.hasNaN() << endl;
    m.col(1)[1] /= 0;
    cout << "m\n" << m << endl;
    cout << "m.hasNaN(): " << m.hasNaN() << endl;

    m.col(1)[1] = sqrt(-1);  // -0. / 0;
    cout << "m\n" << m << endl;
    cout << "m.hasNaN(): " << m.hasNaN() << endl;
    return 0;
}
