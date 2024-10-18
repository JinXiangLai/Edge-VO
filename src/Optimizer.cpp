#include "Optimizer.h"

#include <cfloat>
#include <stdlib.h>
#include <unistd.h>
#include <random>
#include <set>

#include "Config.h"
#include "Eigen/src/Core/Matrix.h"
#include "KeyFrame.h"
#include "Landmark.h"
#include "Utils.h"

#define USE_DT_RESIDUAL // 测试优化算法是否有问题

using namespace std;
using namespace cv;

constexpr double kPriorDepthWeight = 1e8;

Optimizer::Optimizer(const vector<Mat> &dist, const vector<Mat> &dx, const vector<Mat> &dy, shared_ptr<Camera> cam,
    const double lambda, const int maxIte, const bool useInvDepth, const bool onlyPoseUpdate) 
    : lambda_(lambda)
    , dist_(dist)
    , dx_(dx)
    , dy_(dy)
    , maxIte_(maxIte)
    , useInvDepth_(useInvDepth)
    , onlyPoseUpdate_(onlyPoseUpdate)
    , cam_(cam) {
        maxIte_ = config->maxIteration;
    }

Optimizer::Optimizer(shared_ptr<Camera> cam, const double lambda, const int maxIte, const bool useInvDepth, const bool onlyPoseUpdate)
    : lambda_(lambda)
    , maxIte_(maxIte)
    , useInvDepth_(useInvDepth)
    , onlyPoseUpdate_(onlyPoseUpdate)
    , cam_(cam) {
        maxIte_ = config->maxIteration;
    }

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
                double r = BilinearInterpolate(dist, px);
                double J_huber_r = 0;
                r = HuberLoss(r, J_huber_r);

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

            const double w = 1.0; // / p.depthCov_;
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
    // cout << "B - C.T:\n" << B-C.transpose() <<endl;
    Eigen::MatrixXd Dinv(D.rows(), D.cols());
    for(int i = 0; i < pointSize; i+=pointDim) {
        Dinv.block(i, i, pointDim, pointDim).noalias() = D.block(i, i, pointDim, pointDim).inverse();
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
    cout << setprecision(3) << "deltaPose: " << deltaPose.transpose() << endl;


    // 求point增量
    // H * Δx = b ==> C*deltaX_pose + D*deltaX_point = b
    // D*deltaX_point = b - C*deltaX_pose
    // deltaX_point = D.inv * (b - C*deltaX_pose)
    Eigen::VectorXd deltaPoint = Dinv * (new_b.middleRows(poseSize, pointSize) - C * deltaPose);
    // cout << setprecision(3) << "deltaPoint: "<< deltaPoint.transpose() << endl;

    Eigen::VectorXd deltaX(deltaPose.rows() + deltaPoint.rows());
    deltaX.middleRows(0, poseSize) = deltaPose;
    deltaX.middleRows(poseSize, pointSize) = deltaPoint;
    return deltaX;
}

bool Optimizer::ExecuteLMoptimize() {
    vector<Landmark*> debugAllConvergeLandmark = optLandmark_;
    double lastCost = CalculateResidual();

#ifndef USE_DT_RESIDUAL
    MarginalizeOldestKeyFrame();
#else
    // 只需要保留最老帧的信息即可，或者只固定首帧的pose进行优化在debug阶段也是可取的
    MarginalizeOldestKeyFrame();
#endif
    // 如果是使用点-点匹配逻辑的话，那么应该先进行边缘化再转移点的控制权
    // 产生的问题是：那些没有host被边缘化，但是没有target的点不造成影响
    // 那些host被边缘化，但是仍有target的点，可能只剩一个target本身的观测
    RemoveOldestKeyFrame();
    //ShowPointCloud(debugAllConvergeLandmark, optLandmark_, "All vs Opt");

    // 丢失追踪，重新进行
    if (optLandmark_.size() < 10) {
        return false;
    }


    double firstCost = lastCost;
    bool status = false;
    // reset lambda
    lambda_ = 1.0;
    chrono::steady_clock::time_point T1 = chrono::steady_clock::now();
    for(int i = 0; i < maxIte_; ++i) {
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
#ifdef USE_DT_RESIDUAL
        const double cost = ConstructJ_H_b_g();
#else
        const double cost = ConstructJ_H_b_g_byMatch();
#endif

        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        cout << "ConstructJ_H_b_g spend: " << chrono::duration<double>(t2 -t1).count() << " sec." << endl;

        
        Eigen::VectorXd _lambda(H_.rows());
        _lambda.setConstant(lambda_);
        if(Hp_.rows() > 1) {
            if(config->messageLevel == MessageLevel::Debug) {
                cout << "Hp_: [" << Hp_.rows() << "x" << Hp_.cols() << "]" << endl;
                cout << "g_p_: [" << g_p_.rows() << "x1]" << endl;
                cout << "H_: [" << H_.rows() << "x" << H_.cols() << "]" << endl;
                cout << "g_: [" << g_.rows() << "x1]" << endl;
            }
            cout << "Hp_[6x6]: " << setprecision(3) << Hp_.diagonal().head(6).transpose() << endl;
            H_ += Hp_;
            g_ += g_p_;
            //cout << setprecision(3) << "Hp_: " << Hp_.diagonal().transpose() << endl;
            //cout << setprecision(3) << "g_p_: " << g_p_.transpose() << endl;
            cout << "Prior Message Added!!!" << endl;;
        } else {
            _lambda.head(6).setConstant(DBL_MAX); // 首帧的约束足够大
            cout << "Fixed First Frame!!!" << endl;
        }
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
            // 更新先验残差构成信息项
            if(g_p_.rows() > 1) {
                cout << "delta_x: [" << delta_x.rows() << "x1]" << endl;
                UpdatePriorConstraint(delta_x);
            }
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
    cout << "Total Optimize spend " << spendTime << "s\n" << endl;

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
        cout << fixed << "iterate " << i << " times, lastCost | newCost: " << lastCost << " | " << cost << " &lambda: " << lambda_ << endl << endl;

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
        if(lambda_ > 1e20) {
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
    cout << "Total Optimize spend " << spendTime << "s\n" << endl;

    return status;
}

void Optimizer::AddOneKeyFeame(KeyFrame *kf) {
    window_.push_back(kf);
    if(window_.size() > config->maxKFnumInWindow) {
        // margTwc_ = &window_.front()->Twc_;
        // MarginalizeOldestKeyFrame();
    }
}


int Optimizer::TransferLandmarkOwnership() {
    if(window_.size() <= config->maxKFnumInWindow) {
        return 0;
    }
    cout << "window size: " << window_.size() << " begin transfer landmark ownership" << endl;
    KeyFrame *oldest = window_[0];
    int transformNum = 0;
    for(int i = 0; i < oldest->landmark_.size(); ++i) {
        Landmark* p = oldest->landmark_[i];
        if(p == nullptr || p->IsOutOfRange()) {
            continue;
        }

#ifndef USE_DT_RESIDUAL
        // 如果是使用match方式的话，不能删除对landmark的量测权，因为构建先验残差时需要用到
        if(p->host_ == oldest && p->target_.size() == 1) {
            // 只有关于自身的量测，地图点无效了
#else
        // 删除landmark关于oldest的观测，因为p的host不一定是当前帧
        p->target_.erase(oldest);
        if(p->target_.empty()) {
            // host帧观测也存在map容器中，若容器为空，则landmark可以删除
#endif
            p->SetOutOfRange();
        } else if(p->host_ == oldest) {
            const Eigen::Vector3d pw = p->GetPw();
            // p->host_ = nullptr; // 不允许，landmark超过视野不代表其pw是失效的
            // 选一个最新的关键帧，以转移控制权
            //for(int j = window_.size()-1; j >=0; --j) {
            for(int j = 1; j < window_.size(); ++j) {
                KeyFrame *kf = window_[j];
                if(p->target_.count(kf)) {
                    const Eigen::Vector3d pc2 = kf->Tcw_ * pw;
                    if(pc2.z() < config->minDepth || pc2.z() > config->maxDepth) {
                        continue;
                    }
                    Eigen::Vector2d px2 = cam_->Project2PixelPlane(pc2);
                    const double pxError = (px2 - p->target_[kf]).norm();
                    // 像素误差过大，无法转移控制权
                    if(pxError > config->maxTrackProjectPixelError) {
                        continue;
                    }
                    
                    // 可以转移landmark控制权，更新量测、深度及不确定度
                    p->host_ = kf;
                    p->uv_ = p->target_[kf];
                    p->z_ = pc2.z();
                    p->UpdateUncertainty();
                    ++transformNum;
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
    return transformNum;
}


void Optimizer::RemoveOldestKeyFrame() {
    if(window_.size() <= config->maxKFnumInWindow) {
        return;
    }
    cout << "window size: " << window_.size() << " begin remove oldest" << endl;
    KeyFrame *oldest = window_[0];
    window_.erase(window_.begin());
    cout << "window[0]: " << window_[0] << endl;
    // 释放KF，其对应的landmark已经释放
    cout << "[WARNING] kf: " << oldest << " Set out of range flag" << endl;
    // delete oldest; // 不能直接释放，因为其余指向该位置的指针并不会变成nullptr
    oldest->SetOutOfRange();
    historicalKF_.push_back(oldest);
    cout << "historicalKF_.size: " << historicalKF_.size() << endl;

    // 所有观测到该帧的地图点要删除量测
    for(Landmark *p : oldest->landmark_) {
        p->target_.erase(oldest);
        if(p->host_ == oldest) {
            p->SetOutOfRange();
        }
    }
    return;
}

bool Optimizer::SetOptimizeVariables() {
    if(window_.size() < 2) {
        cerr << "Window size: " << window_.size() << " < 2" << endl;
        return false;
    }

    // 添加有效地图点进行优化
    set<Landmark*> ps;
    // 最新帧作为量测帧来更新已有KF的深度
    for(int i = 0; i < window_.size() - 1; ++i) {
        KeyFrame *kf = window_[i];
        vector<Landmark*> &ld = kf->landmark_;
        for(Landmark *p : ld) {
            // 重置FEJ保存的雅可比
            p->ResetFEJ();
            if(p != nullptr && !ps.count(p) && !p->IsOutOfRange() && p->Converge()) {
                ps.insert(p);
            }
        }
    }
    cout << ps.size() << " Landmarks and " << window_.size() << " KFs will participate optimization!!!" << endl;
    optLandmark_.clear();
    optLandmark_ = vector<Landmark*>(ps.begin(), ps.end());
    // 按地址从小到大排序
    //sort(optLandmark_.begin(), optLandmark_.end(), [](Landmark *p1, Landmark*p2){return p1 < p2;});
    return optLandmark_.size() > 100;
}

int Optimizer::SampleUsefulLandmark() {
    vector<vector<Landmark*> > usefulLandmarkEachKF(window_.size()-1);
    map<KeyFrame*, int> kfMapId;
    for(int i = 0; i < window_.size() - 1; ++i) {
        kfMapId.insert({window_[i], i});
    }

    for(int i = 0; i < optLandmark_.size(); ++i) {
        Landmark *p = optLandmark_[i];
        if(p == nullptr || p->IsOutOfRange() || !p->Converge() ) {
            continue;
        }
        KeyFrame *host = p->host_;
        const Eigen::Vector3d pc1 = p->GetPc();
        const Eigen::Vector3d pw = host->Twc_ * pc1;

#ifndef USE_DT_RESIDUAL
        if(!(p->target_.size() == 1) ) {
            // 说明和其他帧有关联，可以构建残差
#else
        if(1) {
            KeyFrame *target = window_.back();
            const Eigen::Vector3d pc2 = target->Tcw_ * pw;
            if(pc2.z() < config->minDepth || pc2.z() > config->maxDepth) {
                continue;
            }
            const Eigen::Vector2d px2 = cam_->Project2PixelPlane(pc2);
            if(!InRange(target->dist_[0], px2.cast<int>()) ) {
                continue;
            }

            double r = BilinearInterpolate(target->dist_[0], px2);
            if(r > config->abnormalProjectResidual) {
                // 残差值异常，判定为离群点
                continue;
            }
#endif

            usefulLandmarkEachKF[kfMapId[host]].push_back(p);
        }
    }

    optLandmark_.clear();
#ifndef USE_DT_RESIDUAL
    // 首帧的所有landmark由于要被边缘化，所以保留
    optLandmark_.insert(optLandmark_.end(), usefulLandmarkEachKF[0].begin(), usefulLandmarkEachKF[0].end());
    // 中间帧需要进行采样关键点，最新帧的不会添加landmark
    for(int i = 1; i < usefulLandmarkEachKF.size(); ++i) {
#else
    for(int i = 0; i < usefulLandmarkEachKF.size(); ++i) {
#endif
        vector<Landmark*> &ps_i = usefulLandmarkEachKF[i];
        if(usefulLandmarkEachKF[i].size() < config->maxActiveLandmarkEachKF) {
            optLandmark_.insert(optLandmark_.end(), ps_i.begin(), ps_i.end());
            continue;
        }
        random_device rd;
        shuffle(ps_i.begin(), ps_i.end(), mt19937(rd() ) );
        optLandmark_.insert(optLandmark_.end(), ps_i.begin(), 
            ps_i.begin()+config->maxActiveLandmarkEachKF);

        //DrawMatch(usefulLandmarkEachKF[i], window_.back(), "Project landmark to last frame");
    }
    
    //DrawMatch(usefulLandmarkEachKF[0], window_.back(), "Project landmark to last frame");
    return optLandmark_.size();
}

double Optimizer::CalculateResidual() {
    double cost = 0;

    vector<Landmark*> usefulLandmark; // 记录能够参与当前优化的边缘点

    for(int i = 0; i < optLandmark_.size(); ++i) {
        Landmark *p = optLandmark_[i];
        KeyFrame *host = p->host_;
        const Eigen::Vector3d pc1 = p->GetPc();
        const Eigen::Vector3d pw = host->Twc_ * pc1;

#ifndef USE_DT_RESIDUAL
        for(const auto &tar : p->target_) {
            // 不能向host投影，TODO: 删除host在target中的观测
            // 根据滑窗性质，只需投影到最后一个KF实现逐步收敛即可
            // if(tar.first == host || tar.first!=window_.back()) {
            if(tar.first == host) {
                continue;
            }
            KeyFrame *target = tar.first;
#else
        if(!(p->target_.size()==1 && p->target_.count(window_.back()))) {
            KeyFrame *target = window_.back();
#endif

            const Eigen::Vector3d pc2 = target->Tcw_ * pw;
            if(pc2.z() < config->minDepth || pc2.z() > config->maxDepth) {
                continue;
            }
            const Eigen::Vector2d px2 = cam_->Project2PixelPlane(pc2);
            if(!InRange(target->dist_[0], px2.cast<int>()) ) {
                continue;
            }

#ifndef USE_DT_RESIDUAL
            const Eigen::Vector2d r = px2 - p->target_[target];
            Eigen::Matrix<double, 1, 2> J_huber_r;
            const double loss = HuberLoss(r, J_huber_r);
            cost += loss;
#else
            double r = BilinearInterpolate(target->dist_[0], px2);
            //if(r > config->abnormalProjectResidual && firstCalculateResidual_) {
            //    // 残差值异常，判定为离群点，已经在采样过程判断了
            //    continue;
            //}
            // 使用胡伯核函数剔除异常残差值
            double J_huber_r = 0;
            r = HuberLoss(r, J_huber_r);
            cost += r;

#endif
        }
    }
    return cost;
}

// TODO： 先不考虑边缘化，而是直接丢弃首帧
void Optimizer::MarginalizeOldestKeyFrame() {
    if(window_.size() <= config->maxKFnumInWindow) {
        return;
    }
    Hp_.resize(1, 1);
    g_p_.resize(1, 1);
    /*********************************************************
    * 注意：VINS-MONO论文中的r_p, Hp分别代表先验残差、先验雅可比，
    * 即 先验约束项 |r_p - Hp * X|^2 <==> |r_p - Jp * X|^2
    * 注意：VINS-MONO论文中，多处出现H矩阵，其均不代表J’*J!!!，而是代表J
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
    
    
    // Stpe: 首先将需要被边缘化的地图点的行和列移动到左上角
    /****************************************************************
    *       T0    T1    d0    d1                       T0   d1   T1   d0
    * T0  T0T0  T1T0  doT0  d1T0                  T0 T0T0 d1T0 T1T0 d0T0
    * T1  T0T1  T1T1  d0T1  d1T1 ==> 移动d1到左上角 d1 T0d1 d1d1 T1d1 d0d1
    * d0  T0d0  T1d0  d0d0  d1d0                  T1 T0T1 d1T1 T1T1 d0T1
    * d1  T0d1  T1d1  d0d1  d1d1                  d0 T0d0 d1d0 T1d0 d0d0
    * 直接操作信息矩阵H_来移动看起来是不可能的，但是我们可以先组织优化变量的顺序，
    * 再计算排序后的H_矩阵，这样直接边缘化左上角就简单了
    *****************************************************************/
    set<Landmark*> margLandmark;
    // TODO: 选择另一种策略移除一帧，类似Landmark的处理方式，将其移到window_[0]再构建信息矩阵H即可
    KeyFrame *oldestKF = window_[0];
    for(int i = 0; i < optLandmark_.size(); ++i) {
        Landmark* p = optLandmark_[i];
        if(p->target_.empty() || (p->target_.size()==1 && p->target_.count(oldestKF))) {
            margLandmark.insert(p);
        }
    }
    vector<Landmark*> sortMargOptLandmark;
    //for(Landmark *p : margLandmark) {
    //    // OK, 这样就实现了将边缘化地图点移到左上角的目的啦！！！
    //    sortMargOptLandmark.push_back(p);
    //}
    for(Landmark *p : optLandmark_) {
        if(!margLandmark.count(p)) {
            sortMargOptLandmark.push_back(p);
        }
    }
    // 根据新的Landmark顺序，构建信息矩阵H_，并保存FEJ
    optLandmark_ = sortMargOptLandmark;
    cout << "total, marg, left landmars: " << optLandmark_.size() << " " << margLandmark.size() 
        << " " << (optLandmark_.size() - margLandmark.size()) << endl;
    // 这里可以实现将线性化点固定在Marginalization时刻
    //ConstructJ_H_b_g();
    ConstructJ_H_b_g_byMatch();

    //ShowPointCloud(sortMargOptLandmark, optLandmark_, "Marg left landmark");

    // Step:接下来计算相关先验Hp, g_p
    const int poseDim = window_[0]->Twc_.Size();
    const int depthDim = 1;
    //const int margDim = poseDim + margLandmark.size() * depthDim;
    // 这里我们直接将最老帧的landmark丢弃不用，只保留边缘化最老帧的信息
    const int margDim = poseDim;
    const int leftDim = H_.cols() - margDim;
    // 使用舒尔补进行边缘化H矩阵，并形成上三角矩阵
    // | I          0 |   | A  B |   | A  B |
    // | -C*A.inv   I | * | C  D | = | 0  ΔA| ==> ΔA = -C*A.inv*B + D
    Eigen::MatrixXd A = H_.block(0, 0, margDim, margDim);
    if(A.diagonal().squaredNorm() < 1.) {
        Hp_.resize(1, 1);
        g_p_.resize(1, 1);
        return;
    }
    Eigen::VectorXd eps(margDim);
    // To avoid A is all Zero，对角线的约束照例说也不应该为0
    eps.setConstant(0);
    A.diagonal() += eps;
    const Eigen::MatrixXd &B = H_.block(0, margDim, margDim, leftDim);
    const Eigen::MatrixXd &C = H_.block(margDim, 0, leftDim, margDim);
    const Eigen::MatrixXd &D = H_.block(margDim, margDim, leftDim, leftDim);
    // TODO: 当边缘化landmark时，可以使用稀疏性求逆
    const Eigen::MatrixXd invA = A.inverse();
    const Eigen::MatrixXd temp = -C * invA;
    Hp_ = -temp*B + D;
    cout << "debug A: " << setprecision(3) << A.diagonal().transpose() << endl
         << "debug B: " << B.diagonal().transpose() << endl
         << "debug D: " << D.diagonal().head(12).transpose() << endl
         << "debug invA: " << invA.diagonal().transpose() << endl
         << "debug temp: " << temp.diagonal().transpose() << endl;
    // | A  B  |   |x1|   | I          0 |   |g1|
    // | 0  ΔA | * |x2| = | -C*A.inv   I | * |g2| ==>
    // TODO: 留下来的状态量X2如果更新，右边的先验残差怎么变呢?
    g_p_ = temp * g_.head(margDim) + g_.tail(leftDim);

    // 易知，先验残差为： |Jp*X - b_p|^2. 其中，Hp_=Jp'*Jp，因此可以得到Jp，g_p_=Jp'*b_p，因此可以得到先验残差b_p(VINS-MONO)
    // 根据G-N方法，展开先验残差项得:
    // (Jp*X)'*(Jp*x) - 2*(Jp*X)'*b_p + b_p'*b_p ==>
    // X'*Jp'*Jp*X - 2*X'*Jp'*b_p + b_p'*bp ==> 极小值点在该式导数为0处，即解满足:
    // Jp'*Jp*X = 2Jp'*bp
    // 如果令 r_p = Jp*Xmarg - b_p，那么当Xnew = Xmarg+ΔX时，先验残差更新为：
    // r_p += Jp*ΔX
    //
    // 
    // 事实上，上式理解为：
    // Hp_*(X-Xmarg) = g_p_, 那么，当Xnew=X+ΔX后，有==>
    // Hp_*(X-Xmarg+ΔX) = g_p_ + Hp_*ΔX，因此，当X更新后，我们需要同步更新
    // g_p_ += Hp_*ΔX

    // 暂时不边缘化点，而是直接丢弃
    //if(!margLandmark.empty()) {
    //    // 构建完H矩阵后，可以从优化地图点中移除marg landmark
    //    optLandmark_.erase(optLandmark_.begin(), optLandmark_.begin() + margLandmark.size());
    //}
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
    cout << "opt variable dim: " << variableDim << endl;
    H_.resize(variableDim, variableDim);
    H_.setZero();
    g_.resize(variableDim);
    g_.setZero();
    double cost = 0;
    int noInrangeNum = 0;
    int depthErrorNum = 0;
    int targetErrorNum = 0;
    int usefulNum = 0;
    // 计算residual & jacobian
    /*********
    *    T0 T1 ... d0 d1 ...
    * r0
    *********/
    const int depthStartCol = window_.size() * poseDim;
    const int resDim = 1;
    int resNum = 0; // 显示当前计算到雅可比的第几行
    // TODO: 有很多landmark不能参与计算，需要将其排除在H及g信息之外，典型的如：
    // opt variable dim: 22902
    // usefulNum, depthErrorNum, targetErrorNum, noInrangeNum, allBadNum: 804 0 23938 1 23939
    for(int i = 0; i < optLandmark_.size(); ++i) {
        Landmark *p = optLandmark_[i];
        KeyFrame *host = p->host_;
        const Eigen::Vector3d pc1 = p->GetPc();
        const Eigen::Vector3d pw = host->Twc_ * pc1;
#ifndef USE_DT_RESIDUAL
        for(const auto &tar : p->target_) {
            // 不能向host投影，TODO: 删除host在target中的观测
            // if(tar.first == host || tar.first!=window_.back()) {
            if(tar.first == host) {
                ++targetErrorNum;
                continue;
            }
            KeyFrame *target = tar.first;
#else
        if(host != window_.back() ) {
            // 我们把所有帧上的深度图投影到最新帧，并优化滑窗内的所有pose
            KeyFrame *target = window_.back();
#endif
            const Eigen::Vector3d pc2 = target->Tcw_ * pw;
            if(pc2.z() < config->minDepth || pc2.z() > config->maxDepth) {
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
            //if(r > kAbnormalResidual) {
            //    // 残差值异常大，认为是离群点
            //    // 只有首次寻找地图点时考虑异常值，后面的目标只是减小损失函数值
            //    continue;
            //}
            // 使用胡伯核函数剔除异常残差值
            double J_huber_r = 0;
            r = HuberLoss(r, J_huber_r);
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
            //const Eigen::Matrix<double, 1, 2> J_res_px2(dx, dy);
            const Eigen::Matrix<double, 1, 2> J_res_px2(J_huber_r * dx, J_huber_r * dy);

            // px2 w.r.t Pc2 [2x3]
            const Eigen::Matrix<double, 2, 3> J_px2_Pc2Norm = p->cam_->K_.block(0, 0, 2, 3);
            Eigen::Matrix<double, 3, 3> J_Pc2Norm_Pc2;
            const double d = 1/pc2.z();
            const double d2 = 1./pow(pc2.z(), 2);
            J_Pc2Norm_Pc2 << d, 0, -pc2.x()*d2,
                            0, d, -pc2.y()*d2,
                            0, 0, 0;
            const Eigen::Matrix<double, 2, 3> J_px2_Pc2 = J_px2_Pc2Norm * J_Pc2Norm_Pc2;
            const Eigen::Matrix<double, 1, 3> J_res_Pc2 = J_res_px2 * J_px2_Pc2;

            // FEJ
            if(!p->J_Pc2_Twc2.count(target)) {
            // if(!p->J_Pc2_Twc2.count(target) || 1) {
                // Pc2 w.r.t Twc2 : Pc2 = Twc2.inv * Pw
                Eigen::Matrix<double, 3, 6> J_Pc2_Twc2; // ------------------------> optimization variable
                const Eigen::Vector3d dt = pw - target->Twc_.t_wb_;
                // Pc2 w.r.t Rwc2
                J_Pc2_Twc2.block(0, 0, 3, 3) = skewSymmetric(target->Tcw_.q_wb_ * dt); // Twc_.q_wb_.inverse()
                // Pc2 w.r.t Pwc2
                J_Pc2_Twc2.block(0, 3, 3, 3) = -target->Tcw_.q_wb_.toRotationMatrix(); // Twc.q_wb.R.transpose()

                // Pc2 w.r.t Pw
                // TODO: 这里也应该要使用首次的Tcw值吧！！！由于target有多帧，所以要保留多个
                const Eigen::Matrix3d J_Pc2_Pw = target->Tcw_.q_wb_.toRotationMatrix();
                
                if(p->J_Pc2_Pw.count(target)) {
                    // just for debug
                    p->J_Pc2_Twc2[target] = J_Pc2_Twc2;
                    p->J_Pc2_Pw[target] = J_Pc2_Pw;
                } else {
                    p->J_Pc2_Twc2.insert({target, J_Pc2_Twc2});
                    p->J_Pc2_Pw.insert({target, J_Pc2_Pw});
                }

                if(p->J_Pw_z.empty()) {
                // if(p->J_Pw_z.empty() || 1) {
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

                    if(!p->J_Pw_z.empty()) {
                        // just for debug
                        p->J_Pw_z.clear();
                        p->J_Pw_Twc1.clear();
                    }
                    p->J_Pw_z.push_back(J_Pw_Pc1 * J_Pc1_z);
                    p->J_Pw_Twc1.push_back(J_Pw_Twc1);
                }
            }

            // Residual w.r.t optimization variables Jacobian
            Eigen::Matrix<double, 1, 6> A1 = J_res_Pc2 * p->J_Pc2_Pw.at(target) * p->J_Pw_Twc1[0]; // J_res_Pw * J_Pw_Twc1;
            if(host == window_[0]) {
                // fixed滑动窗口第一帧
                //A1.setZero();
            }
            Eigen::Matrix<double, 1, 6> A2 =  J_res_Pc2 * p->J_Pc2_Twc2.at(target); // J_res_Pc2 * J_Pc2_Twc2;
            if(target == window_[0]) {
                //A2.setZero();
            }
            const Eigen::Matrix<double, 1, 1> B = J_res_Pc2 * p->J_Pc2_Pw.at(target) * p->J_Pw_z[0]; // J_res_Pw * J_Pw_Pc1 * J_Pc1_z;
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

double Optimizer::ConstructJ_H_b_g_byMatch() {
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
    cout << "opt variable dim: " << variableDim << endl;
    H_.resize(variableDim, variableDim);
    H_.setZero();
    g_.resize(variableDim);
    g_.setZero();
    double cost = 0;
    int noInrangeNum = 0;
    int depthErrorNum = 0;
    int targetErrorNum = 0;
    int usefulNum = 0;
    // 计算residual & jacobian
    /*********
    *    T0 T1 ... d0 d1 ...
    * r0
    *********/
    const int depthStartCol = window_.size() * poseDim;
    constexpr int resDim = 1; // 添加了huberLoss
    int resNum = 0; // 显示当前计算到雅可比的第几行
    // TODO: 有很多landmark不能参与计算，需要将其排除在H及g信息之外，典型的如：
    // opt variable dim: 22902
    // usefulNum, depthErrorNum, targetErrorNum, noInrangeNum, allBadNum: 804 0 23938 1 23939
    for(int i = 0; i < optLandmark_.size(); ++i) {
        Landmark *p = optLandmark_[i];
        KeyFrame *host = p->host_;
        const Eigen::Vector3d pc1 = p->GetPc();
        const Eigen::Vector3d pw = host->Twc_ * pc1;
        
        for(const auto &tar : p->target_) {
            // 不能向host投影，TODO: 删除host在target中的观测
            // if(tar.first == host || tar.first!=window_.back()) {
            if(tar.first == host) {
                ++targetErrorNum;
                continue;
            }
            KeyFrame *target = tar.first;

            const Eigen::Vector3d pc2 = target->Tcw_ * pw;
            if(pc2.z() < config->minDepth || pc2.z() > config->maxDepth) {
                ++depthErrorNum;
                continue;
            }
            const Eigen::Vector2d px2 = cam_->Project2PixelPlane(pc2);
            if(!InRange(target->dist_[0], px2.cast<int>()) ) {
                ++noInrangeNum;
                continue;
            }

            const Eigen::Vector2d r = px2 - p->target_[target];
            Eigen::Matrix<double, 1, 2> J_huber_r(0, 0);
            const double loss = HuberLoss(r, J_huber_r);

            ++usefulNum;
            cost += loss;
            debugKFMapResidualNum[target] += 1;
            resNum += resDim;
            
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
            const Eigen::Matrix<double, 1, 2> J_res_px2 = J_huber_r;

            // px2 w.r.t Pc2 [2x3]
            const Eigen::Matrix<double, 2, 3> J_px2_Pc2Norm = p->cam_->K_.block(0, 0, 2, 3);
            Eigen::Matrix<double, 3, 3> J_Pc2Norm_Pc2;
            const double d = 1/pc2.z();
            const double d2 = 1./pow(pc2.z(), 2);
            J_Pc2Norm_Pc2 << d, 0, -pc2.x()*d2,
                            0, d, -pc2.y()*d2,
                            0, 0, 0;
            const Eigen::Matrix<double, 2, 3> J_px2_Pc2 = J_px2_Pc2Norm * J_Pc2Norm_Pc2;

            // FEJ
            if(!p->J_Pc2_Twc2.count(target)) {
            // if(!p->J_Pc2_Twc2.count(target) || 1) {
                // Pc2 w.r.t Twc2 : Pc2 = Twc2.inv * Pw
                Eigen::Matrix<double, 3, 6> J_Pc2_Twc2; // ------------------------> optimization variable
                const Eigen::Vector3d dt = pw - target->Twc_.t_wb_;
                // Pc2 w.r.t Rwc2
                J_Pc2_Twc2.block(0, 0, 3, 3) = skewSymmetric(target->Tcw_.q_wb_ * dt); // Twc_.q_wb_.inverse()
                // Pc2 w.r.t Pwc2
                J_Pc2_Twc2.block(0, 3, 3, 3) = -target->Tcw_.q_wb_.toRotationMatrix(); // Twc.q_wb.R.transpose()

                // Pc2 w.r.t Pw
                // TODO: 这里也应该要使用首次的Tcw值吧！！！由于target有多帧，所以要保留多个
                const Eigen::Matrix3d J_Pc2_Pw = target->Tcw_.q_wb_.toRotationMatrix();
                
                if(p->J_Pc2_Pw.count(target)) {
                    // just for debug
                    p->J_Pc2_Twc2[target] = J_Pc2_Twc2;
                    p->J_Pc2_Pw[target] = J_Pc2_Pw;
                } else {
                    p->J_Pc2_Twc2.insert({target, J_Pc2_Twc2});
                    p->J_Pc2_Pw.insert({target, J_Pc2_Pw});
                }

                if(p->J_Pw_z.empty()) {
                // if(p->J_Pw_z.empty() || 1) {
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

                    if(!p->J_Pw_z.empty()) {
                        // just for debug
                        p->J_Pw_z.clear();
                        p->J_Pw_Twc1.clear();
                    }
                    p->J_Pw_z.push_back(J_Pw_Pc1 * J_Pc1_z);
                    p->J_Pw_Twc1.push_back(J_Pw_Twc1);
                }
            }

            // Residual w.r.t optimization variables Jacobian
            Eigen::Matrix<double, resDim, 6> A1 = J_res_px2 * J_px2_Pc2 * p->J_Pc2_Pw.at(target) * p->J_Pw_Twc1[0]; // J_res_Pw * J_Pw_Twc1;
            if(host == window_[0]) {
                // fixed滑动窗口第一帧
                //A1.setZero();
            }
            Eigen::Matrix<double, resDim, 6> A2 = J_res_px2 * J_px2_Pc2 * p->J_Pc2_Twc2.at(target); // J_res_Pc2 * J_Pc2_Twc2;
            if(target == window_[0]) {
                //A2.setZero();
            }
            const Eigen::Matrix<double, resDim, 1> B = J_res_px2 * J_px2_Pc2 * p->J_Pc2_Pw.at(target) * p->J_Pw_z[0]; // J_res_Pw * J_Pw_Pc1 * J_Pc1_z;
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
            g_.middleRows(a1j, poseDim) -= A1.transpose() * loss * w;
            g_.middleRows(a2j, poseDim) -= A2.transpose() * loss * w;
            g_.middleRows(bj, depthDim) -= B.transpose() * loss * w; 

        }
    }

    if(config->relativePoseConstraintWeight > 0.) {
        Eigen::MatrixXd H;
        Eigen::VectorXd g;
        ConstructRelativePoseConstraint(H, g);
        H_.block(0, 0, H.rows(), H.cols()) += H;
        g_.middleRows(0, g.rows()) += g;
    }
    cout << "usefulNum, depthErrorNum, targetErrorNum, noInrangeNum, allBadNum: " << usefulNum << " " 
         << depthErrorNum << " " << targetErrorNum << " " << noInrangeNum << " " 
         << (depthErrorNum + targetErrorNum + noInrangeNum) << endl;
    return cost;
}

void Optimizer::ConstructRelativePoseConstraint(Eigen::MatrixXd &H, Eigen::VectorXd &g) {
    if(config->relativePoseConstraintWeight <= 0.) {
        return;
    }
    const int PoseDim = window_[0]->Twc_.Size();
    const int resDim = 6;

    H.resize(window_.size() * PoseDim, window_.size() * PoseDim);
    g.resize(window_.size() * PoseDim);
    H.setZero();
    g.setZero();
    const double w = config->relativePoseConstraintWeight;
    for(int i = 1; i < window_.size(); ++i) {
        KeyFrame *kf1 = window_[i-1];
        KeyFrame *kf2 = window_[i];
        const Pose pTwc1 = kf1->priorTwc_;
        const Pose pTwc2 = kf2->priorTwc_;
        const Pose pTc1c2 = pTwc1.Inverse() * pTwc2;
        const Pose Tc1c2 = kf1->Twc_.Inverse() * kf2->Twc_;
        const Eigen::Quaterniond deltaQ(pTc1c2.q_wb_.inverse() * Tc1c2.q_wb_);
        Eigen::Matrix<double, 6, 1> r_R_P;
        // 实际是 ΔR * Rwc1.inv * Rwc2
        r_R_P.head(3) = LogSO3(deltaQ.toRotationMatrix());
        // 实际是 ΔP - Rwc1.T*Pwc1 + Rwc1.T*Pwc2 
        r_R_P.tail(3) = pTc1c2.t_wb_ - Tc1c2.t_wb_;

        const int aj1 = (i-1) * PoseDim, aj2 = i * PoseDim;
        Eigen::Matrix<double, 6, 6> A1, A2;
        A1.setZero();
        A2.setZero();

        // 使用"BCH近似"之前，需要通过"伴随性质"将扰动量换到右边
        // dLogSO3(ΔR*R1.T*R2) ---> 微分扰动
        // = LogSO3(ΔR*exp(-ε1^)*R1.T*R2) ---> 使用伴随:
        // = LogSO3(ΔR*R1.T*R2 * Exp(-R2.T*R1*ε1)) ---> Exp{小量}，使用BCH近似
        // = [Jr(ΔR*R1.T*R2).inv * -R2.T*R1*ε1] + LogSo3(ΔR*R1.T*R2)
        const Eigen::Matrix3d invJr = InverseRightJacobianSO3(r_R_P.head(3));
        const Eigen::Vector3d dt = kf1->Twc_.t_wb_ - kf2->Twc_.t_wb_;
        const Eigen::Matrix3d R1 = kf1->Twc_.q_wb_.toRotationMatrix();
        const Eigen::Matrix3d R2 = kf2->Twc_.q_wb_.toRotationMatrix();
        
        // LogSO3(ΔR * Rwc1.inv * Rwc2) w.r.t Rwc1
        A1.block(0, 0, 3, 3) = -invJr * R2.transpose() * R1;

        // LogSO3(ΔR * Rwc1.inv * Rwc2) w.r.t Rwc2
        A2.block(0, 0, 3, 3) = invJr;

        // ΔP + Rwc1.T*(Pwc1 - Pwc2) w.r.t Rwc1
        A1.block(3, 0, 3, 3) = skewSymmetric(R1.transpose() * dt);

        // ΔP + Rwc1.T*(Pwc1 - Pwc2) w.r.t Rwc2

        // ΔP + Rwc1.T*(Pwc1 - Pwc2) w.r.t Pwc1
        A1.block(3, 3, 3, 3) = R1.transpose();

        // ΔP + Rwc1.T*(Pwc1 -Pwc2) w.r.t Pwc2
        A2.block(3, 3, 3, 3) = -R1.transpose();

        H.block(aj1, aj1, 6, 6) += A1.transpose() * A1 * w;
        H.block(aj1, aj2, 6, 6) += A1.transpose() * A2 * w;
        H.block(aj2, aj2, 6, 6) += A2.transpose() * A2 * w;
        H.block(aj2, aj1, 6, 6) += A2.transpose() * A1 * w;
        g.middleRows(aj1, 6) -= A1.transpose() * r_R_P * w;
        g.middleRows(aj2, 6) -= A2.transpose() * r_R_P * w;
    }
}

bool Optimizer::SlidingWindowOptimize() {
    cout << "Begin SlidingWindowOptimize!!!" << endl;
    if(!SetOptimizeVariables() ) {
        cerr << "find landmark for optimization num: " << optLandmark_.size() 
            << " too small " << endl;
        // return false;
    }
    int margKFid = SelectOneKF2Marginalization();
    cout << "margKFid: " << margKFid << endl;
   
#ifndef USE_DT_RESIDUAL
     // 转移最老帧点的所有权，以便继续进行优化而非直接边缘化掉
     // 只在跟踪特征点的情况下使用
    const int transformNum = TransferLandmarkOwnership();
    cout << "transformNum: " << transformNum << endl;
#endif

    const int sampleNum = SampleUsefulLandmark();
    cout << "Sample landmark num: " << sampleNum << endl;
    return ExecuteLMoptimize();
}

double Optimizer::HuberLoss(const double residual, double &J_huber_r) {
    double huberLoss = 0;
    const double huberDelta = config->huberDelta;
    if(abs(residual) > huberDelta) {
        // r = δ*(|a|-δ)
        huberLoss = huberDelta * (abs(residual) - 0.5 * huberDelta);
        J_huber_r = residual >= 0? huberDelta : -huberDelta;
    } else {
        // r = 0.5*a^2
        huberLoss = 0.5 * pow(residual, 2);
        J_huber_r = residual;
    }
    return huberLoss;
}

double Optimizer::HuberLoss(const Eigen::Vector2d &residual, Eigen::Matrix<double, 1, 2> &J_huber_r) {
    double huberLoss = 0;
    // TODO: 实现多维变量的胡伯核
    const double huberDelta = config->huberDelta;
    const Eigen::Vector2d delta(huberDelta, huberDelta);
    if(residual.squaredNorm() > delta.squaredNorm()) {
        // r = δ*(|a|-δ)
        huberLoss = delta.dot(residual.cwiseAbs() - 0.5 * delta);
        J_huber_r[0] = residual[0] >= 0 ? huberDelta : -huberDelta;
        J_huber_r[1] = residual[0] >= 0 ? huberDelta : -huberDelta;
    } else {
        // r = 0.5*a^2
        huberLoss = 0.5 * residual.dot(residual);
        J_huber_r = residual.transpose();
    }
    return huberLoss;
}

int Optimizer::SelectOneKF2Marginalization() {
    // Keep the last 2 frame
    if(window_.size() <= config->maxKFnumInWindow) {
        return 1000;
    }
    // 不考虑结构的情况下，移除掉与最新帧观测最少的
    // TODO: 有多帧小于可删除阈值时，考虑删除关键点分布较差的KF
    const int keepLastKFnum = config->keepLastKFnumInWindow;
    vector<double> score(window_.size() - keepLastKFnum, 0);
    for(int i = 0; i < window_.size() - keepLastKFnum; ++i) {
        vector<Landmark*> &ps = window_[i]->landmark_;
        double convergeNum = 0;
        double seenByNewestNum = 0;
        for(Landmark *p : ps) {
            if(p==nullptr || !p->Converge() || p->IsOutOfRange()) {
                continue;
            }
            convergeNum += 1;

#ifndef USE_DT_RESIDUAL
            seenByNewestNum += int(p->target_.size() > 2);      
#else
            const Eigen::Vector3d pc2 = window_.back()->Tcw_ * p->GetPw();
            const Eigen::Vector2d px2 = cam_->Project2PixelPlane(pc2);
            if(InRange(window_.back()->dist_[0], px2.cast<int>()) && 
                window_.back()->dist_[0].at<float>(px2.y(), px2.x()) < config->goodDescriptorDist) {
                seenByNewestNum += 1;
            }
#endif
        }
        score[i] = seenByNewestNum / convergeNum;
    }
    double smallRatio = DBL_MAX;
    int smallId = 0;
    cout << "score: ";
    for(int i = 0; i < score.size(); ++i) {
        cout << score[i] << " ";
        if(score[i] < smallRatio) {
            smallId = i;
            smallRatio = score[i];
        }
    }
    cout << endl;
    KeyFrame *oldest = window_[smallId];
    window_[smallId] = window_[0];
    window_[0] = oldest;
    return smallId;
}

bool Optimizer::UpdateCurrentFrame(KeyFrame *kf2){
    KeyFrame *ref = window_.back();
    optLandmark_.clear();
    for(int i = 0; i < ref->landmark_.size(); ++i) {
        Landmark *kp = ref->landmark_[i];
        if(kp!=nullptr && !kp->IsOutOfRange() && kp->Converge()) {
            optLandmark_.push_back(kp);
        }
    }
    const int maxOptNum = config->maxActiveLandmarkEachKF * 2;
    if(optLandmark_.size() > maxOptNum) {
        random_device rd;
        shuffle(optLandmark_.begin(), optLandmark_.end(), mt19937(rd() ) );
        optLandmark_.erase(optLandmark_.begin()+maxOptNum, optLandmark_.end());
    }

    dist_.clear();
    dist_.push_back(kf2->dist_[0]);
    dx_.clear();
    dx_.push_back(kf2->dx_[0]);
    dy_.clear();
    dy_.push_back(kf2->dy_[0]);
    Pose T12 = ref->Twc_.Inverse() * kf2->Twc_;
    vector<Pose> optPose{T12};
    Optimize(optLandmark_, optPose);
    cout << "cur frame pose diff: " << T12.Inverse() * optPose[0] << endl;
    kf2->SetTwc(ref->Twc_ * optPose[0]);
    return true;
}


double Optimizer::TransformDepthMap2CurrentFrame(KeyFrame *kf2) {
    for(KeyFrame *kf1 : window_) {
        ::TransformDepthMap2CurrentFrame(kf1, kf2, *cam_);
    }

    double convergeNum = 0;
    for(Landmark *lk : kf2->landmark_) {
        if(lk!=nullptr && lk->Converge()) {
            convergeNum += 1;
        }
    }
    return convergeNum / kf2->landmark_.size();
}

void Optimizer::ShowLocalMap() {
    set<Landmark*> ps;
    vector<Pose> vTwc;

    for(int i = 0; i < window_.size(); ++i) {
        // 新插入的最后一个KF未成熟
        KeyFrame *kf = window_[i];
        vTwc.push_back(kf->Twc_);
        for(Landmark *p : kf->landmark_) {
            if(p!=nullptr && !ps.count(p) && p->Converge()) {
                ps.insert(p);
            }
        }
    }
    if(!ps.empty()) {
        ::ShowLocalMap(ps, vTwc);
        cout << "show " << window_.size() << " KFs map points" << endl;
    } else {
        cerr << "wait for local map..." << endl;
    }
}
