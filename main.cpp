//
//  main.cpp
//  SQN
//
//  Created by Jiazheng LIU on 4/18/18.
//  Copyright © 2018 Jiazheng LIU. All rights reserved.
//

#include <iostream>
#include <Eigen/Core>
#include "SQN.h"
#include "SQNparam.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using namespace SQNpp;

class Function
{
private:
    int n;    //dimension
public:
    Function (int& length) : n(length){};
    double Value(std::vector<double>& x, double& z, VectorXd& Omega)
    {
        
    }
    
    VectorXd Gradient(std::vector<double>& x, double& z, VectorXd& Omega)
    {
        
    }
    
    MatrixXd Hessian(std::vector<double>& x, double& z, VectorXd& Omega)
    {
        
    }
    
};



int main(int argc, const char * argv[]) {
    // insert code here...
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    std::cout << "Hello, World!\n";
    return 0;
}
