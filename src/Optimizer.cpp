#include "Optimizer.h"

#include <cstddef>
#include <stdlib.h>
#include <unistd.h>
#include <set>

#include "Config.h"
#include "Eigen/src/Core/Matrix.h"
#include "KeyFrame.h"
#include "Landmark.h"
#include "Utils.h"

// #define TEST // 测试优化算法是否有问题

using namespace std;
using namespace cv;

constexpr double kPriorDepthWeight = 1e8;

Optimizer::Optimizer(const vector<Mat> &dist, const vector<Mat> &dx, const vector<Mat> &dy, std::shared_ptr<Camera> cam,
    const double lambda, const int maxIte, const bool useInvDepth, const bool onlyPoseUpdate) 
    : lambda_(lambda)
    , dist_(dist)
    , dx_(dx)
    , dy_(dy)
    , maxIte_(maxIte)
    , useInvDepth_(useInvDepth)
    , onlyPoseUpdate_(onlyPoseUpdate)
    , cam_(cam) {}

Optimizer::Optimizer(std::shared_ptr<Camera> cam, const double lambda, const int maxIte, const bool useInvDepth, const bool onlyPoseUpdate)
    : lambda_(lambda)
    , maxIte_(maxIte)
    , useInvDepth_(useInvDepth)
    , onlyPoseUpdate_(onlyPoseUpdate)
    , cam_(cam) {}

Eigen::VectorXd Optimizer::CalculateResidual(const vector<Landmark*> &pc1, const vector<Pose> &T12){
    
    constexpr int resDim = 1;
    Eigen::VectorXd res(pc1.size() * T12.size() * resDim + pc1.size());
    // Eigen::VectorXd res(pc1.size() * T12.size() * resDim);
    res.setZero();

    const Camera &cam = *cam_;
    int noInrangeNum = 0;
    double cost = 0;
    for(int i = 0; i < T12.size(); ++i) {
        const Pose T21 = T12[i].Inverse();
        const Mat dist = dist_[i];

        for(int j = 0; j < pc1.size(); ++j) {
            const Eigen::Vector3d pc = T21 * pc1[j]->GetPc();
            const Eigen::Vector2d px = cam.Project2PixelPlane(pc);
            if(InRange(dist, px.cast<int>()) && pc1[j]->z_ > 0) {
                // res[i*pc1.size()*resDim + j] = dist_.at<float>(px.y(), px.x());
                const double r = BilinearInterpolate(dist, px);
                // TODO: 增加异常值鲁棒核函数
                if(r < kAbnormalResidual) {
                    res[i*pc1.size()*resDim + j] = r;
                } else {
                    res[i*pc1.size()*resDim + j] = 1;
                }
                cost += res[i*pc1.size()*resDim + j];
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
                res[startRow+i] = exp(-kPriorDepthWeight / pc1[i]->invZ_);
                continue;
            }
            res[startRow+i] = exp(-kPriorDepthWeight * pc1[i]->z_);
        }
    }

    cout << "residual noInrangeNum: " << noInrangeNum << endl;
    cout << "true Cost: " << fixed << cost << endl;
    return res;
}

Eigen::MatrixXd Optimizer::CalculateJacobian(const vector<Landmark*> &pc1, const vector<Pose> &T12, 
    Eigen::MatrixXd &H, Eigen::VectorXd &b, Eigen::VectorXd &g) {
    
    constexpr int resDim = 1;
    Eigen::MatrixXd J(pc1.size()*T12.size()*resDim + pc1.size(), T12.size()*T12[0].Size() 
        + pc1.size()*pc1[0]->Size());
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
            const Landmark &p = *pc1[j];
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
            const double d = 1/Pc2.z();
            const double d2 = 1./pow(Pc2.z(), 2);
            J_Pc2Norm_Pc2 << d, 0, -Pc2.x()*d2,
                             0, d, -Pc2.y()*d2,
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

            const double w = 1.0 / p.depthCov_;
            J.block(ai, aj, resDim, A.cols()) = A;
            H.block(aj, aj, A.cols(), A.cols()) += A.transpose() * A * w;
            /******** -J.T * b的size为[J.cols() x 1]**************
            * | A.T  C.T  E.T |       | A.T*b1 + C.T*b2 + E.T*b3|
            * | B.T  D.T  F.T | * b = | B.T*b1 + D.T*b2 + F.T*b3|
            *
            *****************************************************/
            g.middleRows(aj, A.cols()) -= A.transpose() * b.middleRows(ai, A.rows()) * w;

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
                H.block(aj, bj, A.cols(), B.cols()) += A.transpose() * B * w;
                H.block(bj, aj, B.cols(), A.cols()) += B.transpose() * A * w;
                H.block(bj, bj, B.cols(), B.cols()) += B.transpose() * B * w;
                g.middleRows(bj, B.cols()) -= B.transpose() * b.middleRows(bi, B.rows()) * w;
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
                B << kPriorDepthWeight / pow(pc1[i]->invZ_, 2) * exp(-kPriorDepthWeight / pc1[i]->invZ_);
            } else {
                B << -kPriorDepthWeight * exp(-kPriorDepthWeight * pc1[i]->z_);
            }
            J.block(bi, bj, 1, 1) = B;
            // 先验z为正的信息，最终还是叠加到了H矩阵和g向量!!!
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

bool Optimizer::ExecuteLMoptimize() {
    double lastCost = -1;
    double firstCost = -1;
    bool status = false;
    // reset lambda
    lambda_ = 1.0;
    chrono::steady_clock::time_point T1 = chrono::steady_clock::now();
    for(int i = 0; i < maxIte_; ++i) {
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        const double cost = ConstructJ_H_b_g();
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        cout << "ConstructJ_H_b_g spend: " << chrono::duration<double>(t2 -t1).count() << " sec." << endl;

        
        Eigen::VectorXd _lambda(H_.rows());
        _lambda.setConstant(lambda_);
        H_.diagonal() += _lambda;
        Eigen::VectorXd delta_x;
        if(!onlyPoseUpdate_) {
            chrono::steady_clock::time_point t3 = chrono::steady_clock::now();
            delta_x = SchurCompleteSolve(H_, g_, window_.size(), optLandmark_.size(), window_[0]->Twc_.Size(), optLandmark_[0]->Size());
            chrono::steady_clock::time_point t4 = chrono::steady_clock::now();
            cout << "SchurCompleteSolve spend: " << chrono::duration<double>(t4 -t3).count() << " sec." << endl;
        } else {
            delta_x = H_.colPivHouseholderQr().solve(g_);
        }
        // cout << setprecision(3) << "delta_x: " << delta_x.transpose() << endl; 
        
        // 保留状态备份
        if(lastCost < 0) {
            lastCost = cost;
            firstCost = lastCost;
        }
        
        vector<Landmark> pcBackup(optLandmark_.size());
        for(int i = 0; i < optLandmark_.size(); ++i) {
            pcBackup[i] = *optLandmark_[i];
        }
        vector<Pose> poseBackup(window_.size());
        for(int i = 0; i < window_.size(); ++i) {
            poseBackup[i] = window_[i]->Twc_;
        }
        
        // 状态更新
        int updateId = 0;
        for(int i = 0; i < window_.size(); ++i) {
            const int startRow = i * window_[0]->Twc_.Size();
            window_[i]->Update(delta_x.middleRows(startRow, 3), delta_x.middleRows(startRow + 3, 3));
        }
        if(!onlyPoseUpdate_) {
            updateId += window_.size() * window_[0]->Twc_.Size();
            for(int i = 0; i < optLandmark_.size(); ++i) {
                optLandmark_[i]->Update(delta_x.middleRows(updateId, optLandmark_[0]->Size())[0], useInvDepth_);
                ++updateId;
            }
        }

        // 判断当前更新是否有效
        const double newCost = CalculateResidual();
        cout << fixed << "iterate " << i << " times, last_cost, new_cost: " << lastCost << " " << newCost 
            << " &lambda: " << lambda_ << endl;

        if(lastCost <= newCost) {
            lambda_ *= 1.8;
            for(int i = 0; i < pcBackup.size(); ++i) {
                *optLandmark_[i] = pcBackup[i];
            }
            for(int i = 0; i < poseBackup.size(); ++i) {
                window_[i]->SetTwc(poseBackup[i]);
            }
        } else {
            lambda_ *= 0.3;
            lastCost = newCost;
        }
        if(newCost < 1e-9) {
            cout << "Congratulations! LM converge!!!" << endl;
            status = true;
            break;
        }
        if(lambda_ > 1e10) {
            cout << fixed << "lambad too large: " << lambda_ << endl;
            break;
        }
        chrono::steady_clock::time_point t5 = chrono::steady_clock::now();
        cout << "LM one iteration spend: " << chrono::duration<double>(t5 -t1).count() << " sec.\n" << endl;

    }
    chrono::steady_clock::time_point T2 = chrono::steady_clock::now();
    const double spendTime = chrono::duration<double>(T2 -T1).count();
    if(!status) {
        cerr << "Reach max iteration time or lambda too large" << endl;
    }
    cout << "First cost | final cost | decrease ratio: " << firstCost << " | " << lastCost << " | "
         << (1. - lastCost/firstCost) * 100 << "%" << endl;
    cout << "Total Optimize spend " << spendTime << "s" << endl;

    return status;
}

bool Optimizer::Optimize(vector<Landmark*> &_pc1, vector<Pose> &T12) {
    vector<Landmark*> pc1;
    pc1.reserve(_pc1.size());
    for(Landmark *p : _pc1) {
        if(p!=nullptr) {
            pc1.push_back(p);
        }
    }
    double lastCost = -1;
    double firstCost = -1;
    bool status = false;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    for(int i = 0; i < maxIte_; ++i) {
        Eigen::VectorXd b;
        Eigen::MatrixXd J = CalculateJacobian(pc1, T12, H_, b, g_);
        Eigen::VectorXd _lambda(H_.rows());
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


        H_.diagonal() += _lambda;
        Eigen::VectorXd delta_x;
        if(!onlyPoseUpdate_) {
            delta_x = SchurCompleteSolve(H_, g_, T12.size(), pc1.size(), T12[0].Size(), pc1[0]->Size());
        } else {
            delta_x = H_.colPivHouseholderQr().solve(g_);
        }
        // cout << setprecision(3) << "delta_x: " << delta_x.transpose() << endl; 
        
        // 保留状态备份
        if(lastCost < 0) {
            lastCost = b.cwiseAbs().sum();
            firstCost = lastCost;
        }
        
        vector<Landmark> pcBackup(pc1.size());
        for(int i = 0; i < pc1.size(); ++i) {
            pcBackup[i] = *pc1[i];
        }
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
                pc1[i]->Update(delta_x.middleRows(updateId, pc1[0]->Size())[0], useInvDepth_);
                ++updateId;
            }
        }

        // 判断当前更新是否有效
        const double cost = CalculateResidual(pc1, T12).cwiseAbs().sum();
        cout << fixed << "iterate " << i << " times, cost: " << lastCost << " &lambda: " << lambda_ << endl << endl;

        if(lastCost <= cost) {
            lambda_ *= 1.8;
            for(int i = 0; i < pcBackup.size(); ++i) {
                *pc1[i] = pcBackup[i];
            }
            T12 = poseBackup;
        } else {
            lambda_ *= 0.3;
            lastCost = cost;
        }
        if(cost < 1e-9) {
            cout << "Congratulations! LM converge!!!" << endl;
            status = true;
            break;
        }
        if(lambda_ > 1e10) {
            cout << fixed << "lambad too large: " << lambda_ << endl;
            break;
        }
    }
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    const double spendTime = chrono::duration<double>(t2 -t1).count();
    if(!status) {
        cerr << "Reach max iteration time or lambda too large" << endl;
    }
    cout << "First cost | final cost | decrease ratio: " << firstCost << " | " << lastCost << " | "
         << (1. - lastCost/firstCost) * 100 << "%" << endl;
    cout << "Total Optimize spend " << spendTime << "s" << endl;

    return status;
}

// TODO： 先不考虑边缘化，而是直接丢弃首帧
void Optimizer::MarginalizeOldestKeyFrame() {
    /*********************************************************
    * 注意：VINS-MONO论文中的r_p, Hp分别代表先验残差、先验雅可比，
    * 即 先验约束项 |r_p - Hp * X|^2 <==> |r_p - Jp * X|^2
    * 注意：VINS-MONO论文中，多处出现H矩阵，其均不代表J’*J!!!
    * 参考为：https://github.com/StevenCui/VIO-Doc/tree/master
    ***********************************************************
    * |A B|   |dx1|   |g1|
    * |C D| * |dx2| = |g2| ==>
    * 将A边缘化掉，得：
    * |E F|   |dx1|   |h1|
    * |0 G| * |dx2| = |h2| ==>
    * G*dx2 = h2, 并且，dx2满足：
    * {我们知道，对于一个线性化的量测方程而言，有：
    * J'*J * dx = -J' * b，
    ******************************************************
    * 令G = J' * J, h2 = -J' * b, 
    * 那么构建先验约束： r = |b - J*X|^2，该式在求解极小值点dx的过程中，
    * 恰好能满足出现： G*dx = h2这一先验约束
    *********************************************************
    * 所以，只被margTwc观测到的地图点，我们不再用它构建方程，直接丢弃
    * 既被margTwc又被其他Twc观测到的地图点，不将其边缘化，而是继续更新
    *******************************************************/
}

void Optimizer::AddOneKeyFeame(KeyFrame *kf) {
    window_.push_back(kf);
    if(window_.size() > kMaxKFnumInWindow) {
        // margTwc_ = &window_.front()->Twc_;
        // MarginalizeOldestKeyFrame();
    }
}

void Optimizer::RemoveOldestKeyFrame() {
    if(window_.size() < kMaxKFnumInWindow) {
        return;
    }
    cout << "window size: " << window_.size() << " begin remove oldest" << endl;
    KeyFrame *oldest = window_[0];
    cout << "oldest: " << oldest << endl;
    window_.erase(window_.begin());
    cout << "window[0]: " << window_[0] << endl;

    for(int i = 0; i < oldest->landmark_.size(); ++i) {
        Landmark* p = oldest->landmark_[i];
        if(p == nullptr || p->IsOutOfRange()) {
            continue;
        }

        // 删除landmark关于oldest的观测，因为p的host不一定是当前帧
        p->target_.erase(oldest);

        if(p->target_.empty()) {
            // host帧观测也存在map容器中，若容器为空，则landmark可以删除
            p->SetOutOfRange();
        } else if(p->host_ == oldest) {
            const Eigen::Vector3d pw = p->GetPw();
            // p->host_ = nullptr; // 不允许，landmark超过视野不代表其pw是失效的
            // 选一个最新的关键帧，以转移控制权
            for(int j = window_.size()-1; j >=0; --j) {
                KeyFrame *kf = window_[j];
                if(p->target_.count(kf)) {
                    const Eigen::Vector3d pc2 = kf->Tcw_ * pw;
                    if(pc2.z() < kMinDepth || pc2.z() > kMaxDepth) {
                        continue;
                    }
                    Eigen::Vector2d px2 = cam_->Project2PixelPlane(pc2);
                    const double pxError = (px2 - p->target_[kf]).norm();
                    // 像素误差过大，无法转移控制权
                    if(pxError > kMaxTrackProjectError) {
                        continue;
                    }
                    
                    // 可以转移landmark控制权，更新量测、深度及不确定度
                    p->host_ = kf;
                    p->uv_ = p->target_[kf];
                    p->z_ = pc2.z();
                    p->UpdateUncertainty();
                    break; // 需要及时撤出哦
                }
            }

            if(p->host_ == oldest) {
                // 未能找到合适的host，删除
                p->SetOutOfRange();
            }

        } else {
            // 控制权在其他帧
            continue;
        }
    }

    // 释放KF，其对应的landmark已经释放
    cout << "[WARNING] kf: " << oldest << " Set out of range flag" << endl;
    // delete oldest; // 不能直接释放，因为其余指向该位置的指针并不会变成nullptr
    oldest->SetOutOfRange();
    cout << "oldest: " << oldest << endl;
    historicalKF_.push_back(oldest);
    cout << "historicalKF_.size: " << historicalKF_.size() << endl;
    return;
}


void Optimizer::ResetOptVariables() {
    // 优化结束后，重置这些标志量
    optLandmark_.clear();
    oldest_ = nullptr;
    newest_ = nullptr;
}

bool Optimizer::SetOptimizeVariables() {
    if(window_.size() < 2) {
        cerr << "Window size: " << window_.size() << " < 2" << endl;
        return false;
    }

    // 添加有效地图点进行优化
    set<Landmark*> ps;
    for(int i = 0; i < window_.size(); ++i) {
        KeyFrame *kf = window_[i];
        vector<Landmark*> &ld = kf->landmark_;
        for(Landmark *p : ld) {
            if(p != nullptr && !ps.count(p) && !p->IsOutOfRange() && p->Converge()) {
                ps.insert(p);
            }
        }
    }
    cout << ps.size() << " Landmarks and " << window_.size() << " KFs will participate optimization!!!" << endl;
    optLandmark_.clear();
    optLandmark_ = vector<Landmark*>(ps.begin(), ps.end());
    // 按地址从小到大排序
    sort(optLandmark_.begin(), optLandmark_.end(), [](Landmark *p1, Landmark*p2){return p1 < p2;});
    
    return optLandmark_.size() > 100;
}

double Optimizer::CalculateResidual() {
    double cost = 0;

    for(int i = 0; i < optLandmark_.size(); ++i) {
        const Landmark *p = optLandmark_[i];
        KeyFrame *host = p->host_;
        const Eigen::Vector3d pc1 = p->GetPc();
        const Eigen::Vector3d pw = host->Twc_ * pc1;
#ifndef TEST
        for(const auto &tar : p->target_) {
            // 不能向host投影，TODO: 删除host在target中的观测
            // 根据滑窗性质，只需投影到最后一个KF实现逐步收敛即可
            if(tar.first == host || tar.first!=window_.back()) {
                continue;
            }
            const KeyFrame *target = tar.first;
#else
        for(const KeyFrame* target : window_) {

            if(target == host) {
                continue;
            }
#endif
            const Eigen::Vector3d pc2 = target->Tcw_ * pw;
            if(pc2.z() < kMinDepth || pc2.z() > kMaxDepth) {
                continue;
            }
            const Eigen::Vector2d px2 = cam_->Project2PixelPlane(pc2);
            if(!InRange(target->dist_[0], px2.cast<int>()) ) {
                continue;
            }

            double r = BilinearInterpolate(target->dist_[0], px2);
            // 使用胡伯核函数剔除异常残差值
            if(r > kAbnormalResidual) {
                // TODO: 检验LM计算比较残差值时是否可以这样计算
                r = 1;
            }
            cost += r;
        }
    }
    return cost;
}

double Optimizer::ConstructJ_H_b_g() {
    // 构建H, g
    // 给出每个KF对应的在H矩阵中的位置
    map<const KeyFrame*, int> kfMapCol;
    map<const KeyFrame*, int> debugKFMapResidualNum;
    for(int i = 0; i < window_.size(); ++i) {
        kfMapCol.insert({window_[i], i * 6});
        debugKFMapResidualNum.insert({window_[i], 0}); // 统计每个图像对应的residual数量
    }
    const int poseDim = window_[0]->Twc_.Size();
    const int depthDim = 1;

    // 或许我们不知道residual，Jacobian的行数，但是H矩阵以及g向量的维度是可知的
    const int variableDim = window_.size() * poseDim + optLandmark_.size() * depthDim;
    H_.resize(variableDim, variableDim);
    H_.setZero();
    g_.resize(variableDim);
    g_.setZero();
    double cost = 0;
    int noInrangeNum = 0;
    int depthErrorNum = 0;
    int targetErrorNum = 0;
    int usefulNum = 0;
    int badNum = 0;
    // 计算residual & jacobian
    /*********
    *    T0 T1 ... d0 d1 ...
    * r0
    *********/
    const int depthStartCol = window_.size() * poseDim;
    const int resDim = 1;
    int resNum = 0; // 显示当前计算到雅可比的第几行
    for(int i = 0; i < optLandmark_.size(); ++i) {
        const Landmark *p = optLandmark_[i];
        KeyFrame *host = p->host_;
        const Eigen::Vector3d pc1 = p->GetPc();
        const Eigen::Vector3d pw = host->Twc_ * pc1;
        cout << "p.tar.size: " << p->target_.size() << endl;
#ifndef TEST
        for(const auto &tar : p->target_) {
            // 不能向host投影，TODO: 删除host在target中的观测
            if(tar.first == host || tar.first!=window_.back()) {
                ++targetErrorNum;
                continue;
            }
            const KeyFrame *target = tar.first;
#else
        for(const KeyFrame* target : window_) {
            if(target == host) {
                ++noInrangeNum;
                continue;
            }
#endif
            const Eigen::Vector3d pc2 = target->Tcw_ * pw;
            if(pc2.z() < kMinDepth || pc2.z() > kMaxDepth) {
                ++depthErrorNum;
                continue;
            }
            const Eigen::Vector2d px2 = cam_->Project2PixelPlane(pc2);
            if(!InRange(target->dist_[0], px2.cast<int>()) ) {
                ++noInrangeNum;
                continue;
            }

            // 经过校验，可以构建residual和jacobian
            double r = BilinearInterpolate(target->dist_[0], px2);
            // 使用胡伯核函数剔除异常残差值
            if(r > kAbnormalResidual) {
                r = 1;
            }
            ++usefulNum;
            cost += r;
            debugKFMapResidualNum[target] += 1;
            resNum += resDim;
            const double dx = BilinearInterpolate(target->dx_[0], px2);
            const double dy = BilinearInterpolate(target->dy_[0], px2);
            /******** 投影过程 ********
            * K.inv * (u1, v1, 1) --> Pc1_norm * z1 --> Twc1 * Pc1 --> Twc2.inv * Pw -->  
            *  Pc2 / z2 -> K * Pc2_norm -> (u2, v2, 1) -> res(u2, v2)
            *
            * res w.r.t (u2, v2) [1x2]
            * (u2, v2) w.r.t Pc2_norm [2x3]
            * Pc2_norm w.r.t Pc2 [3x3]
            * Pc2 w.r.t Twc2 [3x6] ------> optimization variable
            * Pc2 w.r.t Pw [3x3]
            * Pw w.r.t Twc1 [3x6] -------> optimization variable
            * Pw w.r.t Pc1
            * Pc1 w.r.t z1 [3x1] --------> optimization variable
            ************************/

            // res w.r.t (u2, v2) [1x2]
            const Eigen::Matrix<double, 1, 2> J_res_px2(dx, dy);

            // px2 w.r.t Pc2 [2x3]
            const Eigen::Matrix<double, 2, 3> J_px2_Pc2Norm = p->cam_->K_.block(0, 0, 2, 3);
            Eigen::Matrix<double, 3, 3> J_Pc2Norm_Pc2;
            const double d = 1/pc2.z();
            const double d2 = 1./pow(pc2.z(), 2);
            J_Pc2Norm_Pc2 << d, 0, -pc2.x()*d2,
                             0, d, -pc2.y()*d2,
                             0, 0, 0;
            const Eigen::Matrix<double, 2, 3> J_px2_Pc2 = J_px2_Pc2Norm * J_Pc2Norm_Pc2;

            // Pc2 w.r.t Twc2 : Pc2 = Twc2.inv * Pw
            Eigen::Matrix<double, 3, 6> J_Pc2_Twc2; // ------------------------> optimization variable
            const Eigen::Vector3d dt = pw - target->Twc_.t_wb_;
            // Pc2 w.r.t Rwc2
            J_Pc2_Twc2.block(0, 0, 3, 3) = skewSymmetric(target->Tcw_.q_wb_ * dt); // Twc_.q_wb_.inverse()
            // Pc2 w.r.t Pwc2
            J_Pc2_Twc2.block(0, 3, 3, 3) = -target->Tcw_.q_wb_.toRotationMatrix(); // Twc.q_wb.R.transpose()

            // Pc2 w.r.t Pw
            const Eigen::Matrix3d J_Pc2_Pw = target->Tcw_.q_wb_.toRotationMatrix();

            // Pw w.r.t Twc1 : Pw = Twc1 * Pc1 = Rwc1 * pc1 + Pwc1
            Eigen::Matrix<double, 3, 6> J_Pw_Twc1; // ------------------------> optimization variable
            // Pw w.r.t Rwc1
            J_Pw_Twc1.block(0, 0, 3, 3) = -host->Twc_.q_wb_.toRotationMatrix() * skewSymmetric(pc1);
            // Pw w.r.t Pwc1
            J_Pw_Twc1.block(0, 3, 3, 3) = Eigen::Matrix3d::Identity();

            // Pw w.r.t Pc1
            const Eigen::Matrix3d J_Pw_Pc1 = host->Twc_.q_wb_.toRotationMatrix();

            // Pc1 w.r.t z
            const Eigen::Vector3d pc1Norm(p->GetPcNorm());
            Eigen::Vector3d J_Pc1_z{pc1Norm.x(), pc1Norm.y(), 1}; // --------> optimization variable

            const Eigen::Matrix<double, 1, 3> J_res_Pc2 = J_res_px2 * J_px2_Pc2;
            const Eigen::Matrix<double, 1, 3> J_res_Pw = J_res_Pc2 * J_Pc2_Pw;

            // Residual w.r.t optimization variables Jacobian
            Eigen::Matrix<double, 1, 6> A1 = J_res_Pw * J_Pw_Twc1;
            if(host == window_[0]) {
                // fixed滑动窗口第一帧
                A1.setZero();
            }
            Eigen::Matrix<double, 1, 6> A2 = J_res_Pc2 * J_Pc2_Twc2;
            if(target == window_[0]) {
                A2.setZero();
            }
            const Eigen::Matrix<double, 1, 1> B = J_res_Pw * J_Pw_Pc1 * J_Pc1_z;
            const double w = 1.0; // /p->depthCov_;
            const int a1i = resNum, a1j = kfMapCol[host],
                      a2i = resNum, a2j = kfMapCol[target],
                      bi = resNum, bj = depthStartCol + i;
            /********************* 利用稀疏性计算H=J'*J ****************************
            * | A1'|
            * | A2'| * | A1 A2 B |
            * | B' |
            * =
            * | A1'*A1, A1'*A2, A1'*B |
            * | A2'*A1, A2'*A2, A2'*B |
            * | B'*A1,  B'*A2,   B'*B |
            *******************************************************************/
            H_.block(a1j, a1j, poseDim, poseDim) += A1.transpose() * A1 * w;
            H_.block(a1j, a2j, poseDim, poseDim) += A1.transpose() * A2 * w;
            H_.block(a1j, bj, poseDim, depthDim) += A1.transpose() * B * w;

            H_.block(a2j, a1j, poseDim, poseDim) += A2.transpose() * A1 * w;
            H_.block(a2j, a2j, poseDim, poseDim) += A2.transpose() * A2 * w;
            H_.block(a2j, bj, poseDim, depthDim) += A2.transpose() * B * w; 

            H_.block(bj, a1j, depthDim, poseDim) += B.transpose() * A1 * w;
            H_.block(bj, a2j, depthDim, poseDim) += B.transpose() * A2 * w;
            H_.block(bj, bj, depthDim, depthDim) += B.transpose() * B * w;
            /********************* 利用稀疏性计算g=-J'*b ****************************
            * | A1'|       | A1' * b |
            * | A2'| * b = | A2' * b |
            * | B' |       | B'  * b |
            **********************************************************************/
            g_.middleRows(a1j, poseDim) -= A1.transpose() * r * w;
            g_.middleRows(a2j, poseDim) -= A2.transpose() * r * w;
            g_.middleRows(bj, depthDim) -= B.transpose() * r * w; 

        }
    }
    cout << "usefulNum, depthErrorNum, targetErrorNum, noInrangeNum, allBadNum: " << usefulNum << " " 
         << depthErrorNum << " " << targetErrorNum << " " << noInrangeNum << " " 
         << (depthErrorNum + targetErrorNum + noInrangeNum) << endl;
    return cost;
}

bool Optimizer::SlidingWindowOptimize() {
    cout << "Begin SlidingWindowOptimize!!!" << endl;
    if(!SetOptimizeVariables() ) {
        return false;
    }
    return ExecuteLMoptimize();
}

void Optimizer::ShowLocalMap() {
    set<Landmark*> ps;
    for(int i = 0; i < historicalKF_.size(); ++i) {
        // 新插入的最后一个KF未成熟
        KeyFrame *kf = historicalKF_[i];
        for(Landmark *p : kf->landmark_) {
            if(p!=nullptr && !ps.count(p) && p->Converge()) {
                ps.insert(p);
            }
        }
    }
    if(ps.empty()) {
        usleep(100 * 1000);
    } else {
        ShowPointCloud(ps);
    }
}
