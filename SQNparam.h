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

namespace SQNpp {
    
    ///
    /// Parameters to control the Stochastic Quasi-Newton algorithm.
    ///
    template <typename Scalar = double>
    class SQNpp
    {
        
    public:
        /// Size of all available data
        int    N;
        /// input data
        std::vector<std::vector<Scalar> > input_data;
        /// output data
        std::vector<Scalar> output_data;
        ///
        /// Memeory parameter M, controling all the maximal iteration times of
        /// Hessian matrix updating. Noticing the matrix may be very huge,
        /// it is strongly recommended that not set the M too large, normally
        /// the default setting is 10
        int    M;
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
        
    public:
        ///
        /// Constructor for SQN parameters.
        /// Default values for parameters will be set when the object is created.
        ///
        SQNpp()
        {
            M              = 10;
            L              = 100;
            alpha          = Scalar(1);
            b              = 50;
            b_H            = 300;
            m              = 6;
            epsilon        = Scalar(1e-5);
            max_iterations = 0;
            min_step       = Scalar(1e-20);
            max_step       = Scalar(1e+20);
        }
        
        void ReadData() const
        {
            //some read method
            N = output_data.size();
        }
        
        ///
        /// Checking the validity of LBFGS parameters.
        /// An `std::invalid_argument` exception will be thrown if some parameter
        /// is invalid.
        ///
        inline void check_param() const
        {
            if (M <= 0)
                throw std::invalid_argument("'M' must be positive");
            if (M > 100)
                throw std::invalid_argument("'M' Too large! Cannot afford it");
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
