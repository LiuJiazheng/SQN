//
//  main.cpp
//  SQN
//
//  Created by Jiazheng LIU on 4/18/18.
//  Copyright © 2018 Jiazheng LIU. All rights reserved.
//

#include <iostream>
#include <Eigen/Core>
#include <math.h>
#include "SQN.h"
#include "SQNparam.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using namespace SQNpp;

const double EPSILON = 1e-8;

class Function
{
private:
    inline double Entropy(Eigen::VectorXd x, Eigen::VectorXd w)
    {
        double xTw = (x.transpose() * w)(0,0);
        return  1.0 / (1.0 + exp(-1.0 * xTw) + EPSILON) ;
    }
public:
    double Value(Eigen::VectorXd x, Eigen::VectorXd z, VectorXd Omega)
    {
        return  -1.0 * (z(0,0) * log(Entropy(x, Omega)) + (1.0 - z(0,0)) * log(1.0 - Entropy(x, Omega)));
    }
    
    VectorXd Gradient(Eigen::VectorXd x, Eigen::VectorXd z, VectorXd Omega)
    {
        return (Entropy(x,Omega) - z(0,0)) * x;
    }
    
    MatrixXd Hessian(Eigen::VectorXd x, Eigen::VectorXd z, VectorXd Omega)
    {
        long n = Omega.size();
        return MatrixXd::Zero(n, n);
    }
    
    Eigen::VectorXd  Hessian_s(Eigen::VectorXd x , Eigen::VectorXd z, Eigen::VectorXd Omega, Eigen::VectorXd s)
    {
        double xTs = (x.transpose() * s)(0,0);
        return Entropy(x, Omega) * (1 - Entropy(x, Omega)) * xTs * x;
    }
    
};



int main(int argc, const char * argv[]) {
    //set up initial parameter
    SQNpp<double> param;
    SQNreport<double> report(param);
    param.ReadData("/Users/LiuJiazheng/Documents/Optimazation/Data/letter_recognition/OutData.txt");
    SQNsolver<double> slover(param,report);
    
    //create function
    Function BinaryClassfication;
    
    //dimension
    const int n = param.n;
    
    //initial guess
    Eigen::VectorXd Omega(n);
    Omega.setOnes();
    Omega = Omega * 10.0;
    //a space for carry value fx
    double fx;
    
    slover.minimizer(BinaryClassfication, Omega, fx);
    
    return 0;
}
