//
//  FuncTemplate.h
//  SQN
//
//  Created by Jiazheng LIU on 5/29/18.
//  Copyright © 2018 Jiazheng LIU. All rights reserved.
//

#ifndef FuncTemplate_h
#define FuncTemplate_h
#include <Eigen/Core>
#include "SQNparam.h"
using Eigen::VectorXd;
using Eigen::MatrixXd;

class Function
{
public:
    virtual double Value(Eigen::VectorXd x, Eigen::VectorXd z, VectorXd Omega)  = 0;
    
    virtual Eigen::VectorXd Gradient(Eigen::VectorXd x, Eigen::VectorXd z, Eigen::VectorXd Omega)  = 0;
    
    
    virtual Eigen::VectorXd  Hessian_s(Eigen::VectorXd x , Eigen::VectorXd z, Eigen::VectorXd Omega, Eigen::VectorXd s)  =0;
    
    virtual ~Function() {}
    
};



#endif /* FuncTemplate_h */
