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
#include <cassert>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using namespace SQNpp;

const double EPSILON = 1e-8;

class Function
{
private:
    inline double Entropy(Eigen::VectorXd x, Eigen::VectorXd w)
    {
        double xTw = x.dot(w);
        assert(std::isnan(xTw)==false);
        return  1.0 / (1.0 + exp(-1.0 * xTw) + 2 * EPSILON) + EPSILON;
    }
public:
    double Value(Eigen::VectorXd x, Eigen::VectorXd z, VectorXd Omega)
    {
        double result = -1.0 * (z(0,0) * log(Entropy(x, Omega)) + (1.0 - z(0,0)) * log(1.0 - Entropy(x, Omega)));
        if (std::isnan(result))
        {
            double xTw = x.dot(Omega);
            std::cout<<" xTw : "<< xTw <<std::endl;
            std::cout<<"Z : " << z <<std::endl;
            std::cout<<"Entropy : " <<Entropy(x, Omega)<<std::endl;
            std::cout<<" log : "<<log(Entropy(x, Omega))<<std::endl;
            assert(std::isnan(result)==false);
        }
        return result;
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
    SQNheader<double> param;
    if (argc >3)
        throw std::invalid_argument("Wrong number of parameter!");
    param.ReadData(argv[1]);
    SQNreport<double> report(param,argv[2]);
    //param.ReadData("/Users/LiuJiazheng/Documents/Optimazation/Data/letter_recognition/OutData.txt");
    SQNsolver<double> slover(param,report);
    
    //create function
    Function BinaryClassfication;
    
    //dimension
    const int n = param.n;
    
    //initial guess
    Eigen::VectorXd Omega(n);
    Omega.setOnes();
    Omega = Omega * 1.0;
    param.alpha = 0.75;
    param.L = 50;
    param.m = 150;
    //a space for carry value fx
    double fx;
    
    slover.minimizer_enhanced(BinaryClassfication, Omega, fx);
    
    return 0;
}
