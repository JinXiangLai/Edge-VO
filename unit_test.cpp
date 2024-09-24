#include <unistd.h>


#include "Utils.h"
#include "Optimizer.h"

using namespace std;
using namespace cv;

// 利用极线约束去寻找anchor帧与普通帧的匹配以确定匹配特征点
// 得到一个较为准确的深度初值，再与闭环帧执行BA优化

int main(int argc, char** argv) {
    const int v1 = 10, v2 = 19;
    Assert(CalculateDescriptorScore(v1, v2) == 3, "v1^v2 Error!");

    varifyTriangulate();

    cout << "All unit test passed!" << endl;

    int *a = new int(5);
    cout << "a: " << a << endl;
    int *b = a; 
    delete a; // 释放a指向地址的内容
    a = nullptr; // a指向0，但b仍然指向a之前指向的地址
    cout << "null a, b: " << a << " " << b << endl;
    return 0;
}
