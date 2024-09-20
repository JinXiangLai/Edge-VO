#include <stdlib.h>

#include "Optimizer.h"

using namespace std;
using namespace cv;

constexpr double kPriorDepthWeight = 1e8;

Optimizer::Optimizer(const vector<Mat> &dist, const vector<Mat> &dx, const vector<Mat> &dy, const double lambda, 
    const int maxIte, const bool useInvDepth, const bool onlyPoseUpdate) 
    : lambda_(lambda)
    , dist_(dist)
    , dx_(dx)
    , dy_(dy)
    , maxIte_(maxIte)
    , useInvDepth_(useInvDepth)
    , onlyPoseUpdate_(onlyPoseUpdate) {}

Eigen::VectorXd Optimizer::CalculateResidual(const vector<Landmark> &pc1, const vector<Pose> &T12){
    
    constexpr int resDim = 1;
    Eigen::VectorXd res(pc1.size() * T12.size() * resDim + pc1.size());
    // Eigen::VectorXd res(pc1.size() * T12.size() * resDim);
    res.setZero();

    const shared_ptr<Camera> cam = pc1[0].cam_;
    int noInrangeNum = 0;
    for(int i = 0; i < T12.size(); ++i) {
        const Pose T21 = T12[i].Inverse();
        const Mat dist = dist_[i];

        for(int j = 0; j < pc1.size(); ++j) {
            const Eigen::Vector3d pc = T21 * pc1[j].GetPc();
            const Eigen::Vector2d px = cam->Project2PixelPlane(pc);
            if(InRange(dist, px.cast<int>()) && pc1[j].z_ > 0) {
                // res[i*pc1.size()*resDim + j] = dist_.at<float>(px.y(), px.x());
                const double r = BilinearInterpolate(dist, px);
                // TODO: 增加异常值鲁棒核函数
                if(r < kAbnormalResidual) {
                    res[i*pc1.size()*resDim + j] = r;
                } else {
                    res[i*pc1.size()*resDim + j] = 1;
                }
            } else {
                ++noInrangeNum;
                continue;
            }
        }
    }

    // 添加残差，避免深度值z为负
    if(1) {
        const int startRow = pc1.size() * T12.size() * resDim;
        for(int i = 0; i < pc1.size(); ++i) {
            if(useInvDepth_) {
                res[startRow+i] = exp(-kPriorDepthWeight / pc1[i].invZ_);
                continue;
            }
            res[startRow+i] = exp(-kPriorDepthWeight * pc1[i].z_);
        }
    }

    cout << "residual noInrangeNum: " << noInrangeNum << endl;
    return res;
}

Eigen::MatrixXd Optimizer::CalculateJacobian(const vector<Landmark>&pc1, const vector<Pose> &T12, 
    Eigen::MatrixXd &H, Eigen::VectorXd &b, Eigen::VectorXd &g) {
    
    constexpr int resDim = 1;
    Eigen::MatrixXd J(pc1.size()*T12.size()*resDim + pc1.size(), T12.size()*T12[0].Size() + pc1.size()*pc1[0].Size());
    // Eigen::MatrixXd J(pc1.size()*T12.size()*resDim, T12.size()*T12[0].Size() + pc1.size()*pc1[0].Size());

    if(onlyPoseUpdate_) {
        J.resize(J.rows(), T12.size() * T12[0].Size());
    }
    J.setZero();
    H.resize(J.cols(), J.cols());
    H.setZero();
    b = CalculateResidual(pc1, T12);
    g.resize(J.cols());
    g.setZero();

    /******** 投影过程 ********
    * K.inv * (u1, v1, 1) -> Pc1_norm * z1 -> T12.inv * Pc1 -> Pc2 / z2 -> K * Pc2_norm -> (u2, v2, 1) -> res(u2, v2)
    * res w.r.t (u2, v2) [1x2]
    * (u2, v2) w.r.t Pc2_norm [2x3]
    * Pc2_norm w.r.t Pc2 [3x3]
    * Pc2 w.r.t T12 [3x6] ------> optimization variable
    * Pc2 w.r.t Pc1 [3x3]
    * Pc1 w.r.t z1 [3x1] -------> optimization variable
    ************************/
    for(int i = 0; i < T12.size(); ++i) {
        const int poseStartCol = T12[0].Size() * i;
        int pointStartCol = T12.size() * T12[0].Size();

        const Pose T21 = T12[i].Inverse();
        const Mat dxMat = dx_[i];
        const Mat dyMat = dy_[i];

        for(int j = 0; j < pc1.size(); ++j) {
            const Landmark &p = pc1[j];
            const Eigen::Vector3d Pc1 = p.GetPc();
            const Eigen::Vector3d Pc2 = T21 * Pc1;
            const Eigen::Vector2d px2 = p.cam_->Project2PixelPlane(Pc2);
            if(!InRange(dxMat, px2.cast<int>())) {
                continue;
            }
            const double dx = BilinearInterpolate(dxMat, px2);
            const double dy = BilinearInterpolate(dyMat, px2);
            
            // res w.r.t px2 [1x2]
            const Eigen::Matrix<double, 1, 2> J_res_px2(dx, dy);

            // px2 w.r.t Pc2 [2x3]
            Eigen::Matrix<double, 2, 3> J_px2_Pc2Norm = p.cam_->K_.block(0, 0, 2, 3);
            Eigen::Matrix<double, 3, 3> J_Pc2Norm_Pc2;
            J_Pc2Norm_Pc2 << 1/Pc2[2], 0, -Pc2[0]/pow(Pc2[2], 2),
                            0, 1/Pc2[2], -Pc2[1]/pow(Pc2[2], 2),
                            0, 0, 0;
            Eigen::Matrix<double, 2, 3> J_px2_Pc2 = J_px2_Pc2Norm * J_Pc2Norm_Pc2;

            // Pc2 w.r.t T12 [3x6]
            Eigen::Matrix<double, 3, 6> J_Pc2_T12 = Eigen::Matrix<double, 3, 6>::Zero();
            const Eigen::Vector3d dt = Pc1 - T12[i].t_wb_;
            // * Pc2 w.r.t R12
            J_Pc2_T12.block(0, 0, 3, 3) = skewSymmetric(T21.q_wb_ * dt);
            // * Pc2 w.r.t t12
            J_Pc2_T12.block(0, 3, 3, 3) = -T21.q_wb_.toRotationMatrix();

            // Pc2 w.r.t Pc1 [3x3]
            Eigen::Matrix<double, 3, 3> J_Pc2_Pc1 = T21.q_wb_.toRotationMatrix();

            // Pc1 w.r.t z1 [3x1]
            const Eigen::Vector3d pc1Norm(p.GetPcNorm());      
            Eigen::Matrix<double, 3, 1> J_Pc1_z1{pc1Norm.x(), pc1Norm.y(), 1};
            // 使用逆深度表示
            if(useInvDepth_) {
                const double d = pow(p.invZ_, 2);
                J_Pc1_z1 << -pc1Norm.x()/d, -pc1Norm.y()/d, -1/d;
            }

            // 给整体雅可比矩阵赋值
            // TODO： 不需要Fixed pose，只要相对Pose准确
            // H = J'*J, g = -J'*b;
            const int ai = i * pc1.size() + j, aj = poseStartCol;
            const int bi = i * pc1.size() + j, bj = pointStartCol + j;
            Eigen::MatrixXd A = J_res_px2 * J_px2_Pc2 * J_Pc2_T12;
            // A.setZero();
            Eigen::MatrixXd B = J_res_px2 * J_px2_Pc2 * J_Pc2_Pc1 * J_Pc1_z1;
            J.block(ai, aj, resDim, A.cols()) = A;
            H.block(aj, aj, A.cols(), A.cols()) += A.transpose() * A;
            /******** -J.T * b的size为[J.cols() x 1]**************
            * | A.T  C.T  E.T |       | A.T*b1 + C.T*b2 + E.T*b3|
            * | B.T  D.T  F.T | * b = | B.T*b1 + D.T*b2 + F.T*b3|
            *
            *****************************************************/
            g.middleRows(aj, A.cols()) -= A.transpose() * b.middleRows(ai, A.rows());

            if(!onlyPoseUpdate_) {
                J.block(bi, bj, 1, 1) = B;
                /******** 利用分块及稀疏矩阵性质直接计算H矩阵 ********
                * 否则，H=J.T * J由于没有利用到稀疏性，计算量将异常大
                * | A.T, C.T, E.T |   | A, B|
                * | B.T, D.T, F.T | * | C, D|
                *                     | E, F| = 
                * | A.T*A + C.T*C + E.T*E,  A.T*B + C.T*D + E.T*F |
                * | B.T*A + D.T*C + F.T*E,  B.T*B + D.T*D + F.T*F |
                * 观察D、E矩阵块的变化规律，可以写出如下的等式
                */
                H.block(aj, bj, A.cols(), B.cols()) += A.transpose() * B;
                H.block(bj, aj, B.cols(), A.cols()) += B.transpose() * A;
                H.block(bj, bj, B.cols(), B.cols()) += B.transpose() * B;
                g.middleRows(bj, B.cols()) -= B.transpose() * b.middleRows(bi, B.rows());
            }
        }
    }

    // 添加深度值z非负雅可比
    if(1) {
        const int startRow = pc1.size() * T12.size() * resDim;
        int pointStartCol = T12.size() * T12[0].Size();
        for(int i = 0; i < pc1.size(); ++i) {
            const int bi = startRow+i, bj = pointStartCol+i;
            Eigen::MatrixXd B(1, 1);
            if(useInvDepth_) {
                B << kPriorDepthWeight / pow(pc1[i].invZ_, 2) * exp(-kPriorDepthWeight / pc1[i].invZ_);
            } else {
                B << -kPriorDepthWeight * exp(-kPriorDepthWeight * pc1[i].z_);
            }
            J.block(bi, bj, 1, 1) = B;
            H.block(bj, bj, B.cols(), B.cols()) += B.transpose() * B;
            g.middleRows(bj, B.cols()) -= B.transpose() * b.middleRows(bi, B.rows());
        }
    }

    return J;
}

Eigen::VectorXd Optimizer::SchurCompleteSolve(const Eigen::MatrixXd &H, const Eigen::VectorXd &b, const int poseNum, const int pointNum, 
    const int poseDim, const int pointDim) {
    /***
    *     T   p...
    * T   A   B
    * p   C   D
    * ...
    ***/
    // 对B进行边缘化，左乘形成上三角矩阵
    // | I          0 |   | A  B |   | A  B |
    // | -C*A.inv   I | * | C  D | = | 0  ΔA| ==> ΔA = -C*A.inv*B + D
    // 对C进行边缘化，右乘形成下三角矩阵
    // | A  B |   | I  -A.inv*B |   | A  0 |
    // | C  D | * | 0       I   | = | C  ΔA| = H' ==> 用来求pose
    // 可以得到:
    // | I        0 |   | A  B |   | I  -A.inv*B |   | A  0 |
    // |-C*A.inv  I | * | C  D | * | 0      I    | = | 0  ΔA| = H'
    
    // 自己可以构建舒尔补，左乘形成下三角矩阵，先求解pose增量，再求解point增量
    // | I  -B*D.inv |   | A  B |   | A-B*D.inv*C  0 |
    // | 0     I     | * | C  D | = |    C         D |
    // H -= lambdaMatrix;
    const int poseSize = poseNum * poseDim;
    const int pointSize = H.cols() - poseSize;

    const Eigen::MatrixXd &A = H.block(0, 0, poseSize, poseSize);
    const Eigen::MatrixXd &B = H.block(0, poseSize, poseSize, pointSize);
    const Eigen::MatrixXd &C = H.block(poseSize, 0, pointSize, poseSize);
    const Eigen::MatrixXd &D = H.block(poseSize, poseSize, pointSize, pointSize);
    // cout << "B - C.T:\n" << B-C.transpose() <<std::endl;
    Eigen::MatrixXd Dinv(D.rows(), D.cols());
    for(int i = 0; i < pointSize; i+=pointDim) {
        Dinv.block(i, i, pointDim, pointDim) = D.block(i, i, pointDim, pointDim).inverse();
    }
    const Eigen::MatrixXd E = -B * Dinv;
    Eigen::MatrixXd leftMatrix(H.rows(), H.cols());
    leftMatrix.block(0, 0, poseSize, poseSize).setIdentity();
    leftMatrix.block(0, poseSize, poseSize, pointSize) = E;
    leftMatrix.block(poseSize, 0, pointSize, poseSize).setZero();
    leftMatrix.block(poseSize, poseSize, pointSize, pointSize).setIdentity();

    // 求pose增量
    Eigen::MatrixXd newA = A + E * C;
    Eigen::VectorXd new_b = leftMatrix * b;
    Eigen::VectorXd deltaPose = newA.inverse() * (new_b).head(poseSize);
    cout << setprecision(3) << "deltaPose: " << deltaPose.transpose() << std::endl;



    // 求point增量
    // H * Δx = b ==> C*deltaX_pose + D*deltaX_point = b
    // D*deltaX_point = b - C*deltaX_pose
    // deltaX_point = D.inv * (b - C*deltaX_pose)
    Eigen::VectorXd deltaPoint = Dinv * (new_b.middleRows(poseSize, pointSize) - C * deltaPose);
    // std::cout << setprecision(3) << "deltaPoint: "<< deltaPoint.transpose() << std::endl;

    Eigen::VectorXd deltaX(deltaPose.rows() + deltaPoint.rows());
    deltaX.middleRows(0, poseSize) = deltaPose;
    deltaX.middleRows(poseSize, pointSize) = deltaPoint;
    return deltaX;
}

bool Optimizer::Optimize(vector<Landmark> &pc1, vector<Pose> &T12) {
    double lastCost = -1;
    double firstCost = -1;
    bool status = false;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    for(int i = 0; i < maxIte_; ++i) {
        Eigen::MatrixXd H;
        Eigen::VectorXd b, g;
        Eigen::MatrixXd J = CalculateJacobian(pc1, T12, H, b, g);
        Eigen::VectorXd _lambda(H.rows());
        _lambda.setConstant(lambda_);
        
        // debug, 判断H, g计算的正确性
        // Eigen::MatrixXd _H = J.transpose() * J;
        // Eigen::VectorXd _g = -J.transpose() * b;
        // const Eigen::MatrixXd dH = H-_H;
        // const Eigen::VectorXd dg = g-_g;
        // cout << setprecision(3) << "H-_H:\n " << dH.diagonal().transpose() << endl << endl;
        // cout << setprecision(3) << "g-_g:\n " << dg.transpose() << endl << endl;
        // cout << "dH, dg norm: " << dH.norm() << " " << dg.norm() << endl;
        // H = _H;
        // g = _g;


        H.diagonal() += _lambda;
        Eigen::VectorXd delta_x;
        if(!onlyPoseUpdate_) {
            delta_x = SchurCompleteSolve(H, g, T12.size(), pc1.size(), T12[0].Size(), pc1[0].Size());
        } else {
            delta_x = H.colPivHouseholderQr().solve(g);
        }
        // cout << setprecision(3) << "delta_x: " << delta_x.transpose() << endl; 
        
        // 保留状态备份
        if(lastCost < 0) {
            lastCost = b.norm();
            firstCost = lastCost;
        }
        const vector<Landmark> pcBackup = pc1;
        const vector<Pose> poseBackup = T12;
        
        // 状态更新
        int updateId = 0;
        for(int i = 0; i < T12.size(); ++i) {
            const int startRow = i * T12[i].Size();
            T12[i].Update(delta_x.middleRows(startRow, 3), delta_x.middleRows(startRow + 3, 3));
        }
        if(!onlyPoseUpdate_) {
                updateId += T12.size() * T12[0].Size();
            for(int i = 0; i < pc1.size(); ++i) {
                pc1[i].Update(delta_x.middleRows(updateId, pc1[0].Size())[0], useInvDepth_);
                ++updateId;
            }
        }

        // 判断当前更新是否有效
        const double cost = CalculateResidual(pc1, T12).norm();
        cout << fixed << "iterate " << i << " times, cost: " << lastCost << " &lambda: " << lambda_ << endl << endl;

        if(lastCost <= cost) {
            lambda_ *= 1.8;
            pc1 = pcBackup;
            T12 = poseBackup;
        } else {
            lambda_ *= 0.3;
            lastCost = cost;
        }
        if(cost < 1e-9) {
            cout << "Congratulations! LM converge!!!" << endl;
            cout << "First cost | final cost: " << firstCost << " | " << lastCost << endl;
            status = true;
        }
    }
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    const double spendTime = chrono::duration<double>(t2 -t1).count();
    if(!status) {
        cerr << "Reach max iteration time." << endl;
    }
    cout << "First cost | final cost: " << firstCost << " | " << lastCost << endl;
    cout << "Total Optimize spend " << spendTime << "s" << endl;

    return status;
}
