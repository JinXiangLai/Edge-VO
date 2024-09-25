#include <unistd.h>
#include <thread>

#include "Pose.h"
#include "Utils.h"
#include "Optimizer.h"

using namespace std;
using namespace cv;

// 利用极线约束去寻找anchor帧与普通帧的匹配以确定匹配特征点
// 得到一个较为准确的深度初值，再与闭环帧执行BA优化

viz::Viz3d window("Point Cloud Viewer"); // 这个窗口一直在
cv::Affine3d viewPose;


void ShowPointCloud() {
    // viz::Viz3d window("Point Cloud Viewer"); // 放在这里可以
    window.setViewerPose(viewPose);
    vector<Point3d> points;
    vector<Vec3b> colors;
    // 显示一个长方体点云
    for(float i = 0; i < 10; i+=0.1) {
        for(float j = 0; j < 10; j+=0.1) {
            points.push_back({i, j, double(rand()%100)});
            colors.push_back({255, 255, 255});
        }
    }

    viz::WCloud cloud(points, colors);
    // cloud.setColor(cv::viz::Color::green());
    // cloud.setSize(5);
 
    // 显示点云
    window.showWidget("PointCloud", cloud);
    // 运行事件循环，使窗口响应用户输入
    // window.spinOnce(1);
    window.spin();
    // window.close();
    window.removeAllWidgets();
    viewPose = window.getViewerPose();
}

void Run() {
    while(1) {
        ShowPointCloud();
        cout << "happy\n";
        sleep(1);
    }
}

int main(int argc, char** argv) {
    const int v1 = 10, v2 = 19;
    Assert(CalculateDescriptorScore(v1, v2) == 3, "v1^v2 Error!");

    varifyTriangulate();

    int *a = new int(5);
    cout << "a: " << a << endl;
    int *b = a; 
    delete a; // 释放a指向地址的内容
    a = nullptr; // a指向0，但b仍然指向a之前指向的地址
    cout << "null a, b: " << a << " " << b << endl;

    Eigen::Vector3d t_c1c2{0.0, 0.0, -3.5}; 
    const Pose Tc1c2 = ConvertRPYandPostion2Pose({0, 2, 3}, t_c1c2, kDeg2Rad);
    const Pose Tidentity = Tc1c2 * Tc1c2.Inverse();
    cout << "Tidentity: " << Tidentity << endl;
    Assert(Tidentity.t_wb_.isApprox(Eigen::Vector3d::Zero()) && 
        Tidentity.q_wb_.isApprox(Eigen::Quaterniond::Identity()), "Inverse operate Error!");
    cout << "Tidentity: " << Tidentity << endl;

    // ShowPointCloud();
    // thread th(Run);
    // th.join();
    // th.detach();
    
    cout << "All unit test passed!" << endl;

    return 0;
}
