//
//  SQNparam.h
//  SQN
//
//  Created by Jiazheng LIU on 4/18/18.
//  Copyright © 2018 Jiazheng LIU. All rights reserved.
//

#ifndef SQNparam_h
#define SQNparam_h

#include <Eigen/Core>
#include <stdexcept>  // std::invalid_argument
#include <vector>
#include <sstream>
#include <fstream>
#include <string>

inline std::vector<std::string> split(std::string str, char delimiter) {
    std::vector<std::string> internal;
    std::stringstream ss(str); // Turn the string into a stream.
    std::string tok;
    
    while(getline(ss, tok, delimiter)) {
        internal.push_back(tok);
    }
    
    return internal;
}


enum LINE_SEARCH_ALGORITHM
{
    ///
    /// Backtracking method with the Armijo condition.
    /// The backtracking method finds the step length such that it satisfies
    /// the sufficient decrease (Armijo) condition,
    /// \f$f(x + a \cdot d) \le f(x) + \beta' \cdot a \cdot g(x)^T d\f$,
    /// where \f$x\f$ is the current point, \f$d\f$ is the current search direction,
    /// \f$a\f$ is the step length, and \f$\beta'\f$ is the value specified by
    /// \ref LBFGSParam::ftol. \f$f\f$ and \f$g\f$ are the function
    /// and gradient values respectively.
    ///
    LBFGS_LINESEARCH_BACKTRACKING_ARMIJO = 1,
    
    ///
    /// The backtracking method with the defualt (regular Wolfe) condition.
    /// An alias of `LBFGS_LINESEARCH_BACKTRACKING_WOLFE`.
    ///
    LBFGS_LINESEARCH_BACKTRACKING = 2,
    
    ///
    /// Backtracking method with regular Wolfe condition.
    /// The backtracking method finds the step length such that it satisfies
    /// both the Armijo condition (`LBFGS_LINESEARCH_BACKTRACKING_ARMIJO`)
    /// and the curvature condition,
    /// \f$g(x + a \cdot d)^T d \ge \beta \cdot g(x)^T d\f$, where \f$\beta\f$
    /// is the value specified by \ref LBFGSParam::wolfe.
    ///
    LBFGS_LINESEARCH_BACKTRACKING_WOLFE = 2,
    
    ///
    /// Backtracking method with strong Wolfe condition.
    /// The backtracking method finds the step length such that it satisfies
    /// both the Armijo condition (`LBFGS_LINESEARCH_BACKTRACKING_ARMIJO`)
    /// and the following condition,
    /// \f$\vert g(x + a \cdot d)^T d\vert \le \beta \cdot \vert g(x)^T d\vert\f$,
    /// where \f$\beta\f$ is the value specified by \ref LBFGSParam::wolfe.
    ///
    LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE = 3
    };


namespace SQNpp {
    
    ///
    /// Parameters to control the Stochastic Quasi-Newton algorithm.
    ///
    template <typename Scalar = double>
    class SQNheader
    {
        
    public:
        /// Size of all available data
        int    N;
        /// input data
        std::vector<std::vector<Scalar> > input_data;
        /// output data
        std::vector<Scalar> output_data;
        ///
        /// direction updating time windows L. In SQN method, there is a paremeter
        /// to control the frequency of updating search direction, and that is L
        int    L;
        ///
        /// x_{k+1} = x_k + alpha * direction
        Scalar alpha;
        ///
        /// The size of batch for computing gradient descent.
        int    b;
        ///
        /// The size of batch for computing Hessian
        int   b_H;
        ///
        /// updating counter t
        ///
        int    m;
        ///
        /// Tolerance for convergence test.
        /// This parameter determines the accuracy with which the solution is to
        /// be found. A minimization terminates when
        /// \f$||g|| < \epsilon * \max(1, ||x||)\f$,
        /// where ||.|| denotes the Euclidean (L2) norm. The default value is
        /// \c 1e-5.
        ///
        /// Variable Dimension n
        int    n;
        ///
        ///
        Scalar epsilon;
        ///
        /// The maximum number of iterations.
        /// The optimization process is terminated when the iteration count
        /// exceedes this parameter. Setting this parameter to zero continues an
        /// optimization process until a convergence or error. The default value
        /// is \c 0.
        ///
        int    max_iterations;
        ///
        /// The minimum step length allowed in the line search.
        /// The default value is \c 1e-20. Usually this value does not need to be
        /// modified.
        ///
        Scalar min_step;
        ///
        /// The maximum step length allowed in the line search.
        /// The default value is \c 1e+20. Usually this value does not need to be
        /// modified.
        ///
        Scalar max_step;
        
        ///
        /// The line search algorithm.
        /// This parameter specifies the line search algorithm that will be used
        /// by the LBFGS routine. The default value is `LBFGS_LINESEARCH_BACKTRACKING_ARMIJO`.
        ///
        int    linesearch;
        ///
        /// The maximum number of trials for the line search.
        /// This parameter controls the number of function and gradients evaluations
        /// per iteration for the line search routine. The default value is \c 20.
        ///
        int    max_linesearch;
        ///
        
    public:
        ///
        /// Constructor for SQN parameters.
        /// Default values for parameters will be set when the object is created.
        ///
        SQNheader()
        {
            L              = 100;
            alpha          = Scalar(1);
            b              = 50;    //50
            b_H            = 300;    //300
            m              = 6;
            epsilon        = Scalar(1e-5);
            max_iterations = int(1e4);
            min_step       = Scalar(1e-20);
            max_step       = Scalar(1e+20);
            linesearch     = LBFGS_LINESEARCH_BACKTRACKING_ARMIJO;
            max_linesearch = 20;
        }
        
        void ReadData(std::string FileName,char delimiter = ' ') 
        {
            //some read method
            std::string line;
            std::ifstream myfile (FileName);
            if (myfile.is_open())
            {
                while ( std::getline (myfile,line) )
                {
                    std::vector<std::string> templine = split(line,delimiter);
                    output_data.push_back(double(std::stod(*templine.begin())));
                    std::vector<double> tempIn;
                    for (auto it = templine.begin() + 1 ; it != templine.end(); ++it)
                        tempIn.push_back(double(std::stod(*it)));
                    input_data.push_back(tempIn);
                }
                myfile.close();
            }
            else
                throw std::invalid_argument(" Input file does not exist!");
            N = output_data.size();
            n = input_data[0].size();
        }
        
        ///
        /// Checking the validity of LBFGS parameters.
        /// An `std::invalid_argument` exception will be thrown if some parameter
        /// is invalid.
        ///
        inline void check_param() const
        {
            if (b <= 0)
                throw std::invalid_argument("'b' must be positive");
            if (b_H <= 0)
                throw std::invalid_argument("'b_H' must be positive");
            if (alpha <= 0)
                throw std::invalid_argument("decent rate 'alpha' must be positive");
            if(m <= 0)
                throw std::invalid_argument("'m' must be positive");
            if(epsilon <= 0)
                throw std::invalid_argument("'epsilon' must be positive");
            if(max_iterations < 0)
                throw std::invalid_argument("'max_iterations' must be non-negative");
            if(min_step < 0)
                throw std::invalid_argument("'min_step' must be positive");
            if(max_step < min_step )
                throw std::invalid_argument("'max_step' must be greater than 'min_step'");
            if (N <= 0)
                throw std::invalid_argument("Data set must not be empty!");
        }
    };
    
    
} // namespace SQNpp

#endif /* SQNparam_h */
