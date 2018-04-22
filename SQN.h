//
//  SQN.h
//  SQN
//
//  Created by Jiazheng LIU on 4/18/18.
//  Copyright © 2018 Jiazheng LIU. All rights reserved.
//

#ifndef SQN_h
#define SQN_h

#include <Eigen/Core>
#include "SQNparam.h"
#include <random>
#include "SQNreport.h"
#include <stdlib.h>     /* srand, rand */
#include <iostream>

namespace SQNpp {
    template <typename Scalar>
    class SQNsolver{
    private:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        
        //std::default_random_engine random_generator;
        
        const SQNpp<Scalar>&   param;  // Parameters to control the SQN algorithm
        Scalar rate;
        Vector omega;        //direction by gradient descent
        Vector omega_bar ;   //direction for updadting matrix
        Vector prev_omega_bar ;
        Vector s;            //w_t - w_{t-1} difference of direction
        Vector y;            //Nabla_Square_F * s
        Vector Nabla_F;       //Nabla_F
        Vector Prev_Nabla_F;   //used for testing whether convergence
        Matrix Hessian_t ;     // second-order optimal condition: Hessian Matrix
        std::vector<Vector> History_s;
        std::vector<Vector> History_y;
        SQNreport<Scalar>   Report;    //for generating report
        
        inline void reset(int n)
        {
            const int m = param.m;
            rate = Scalar(1);
            omega.resize(n);
            omega_bar.resize(n);
            prev_omega_bar.resize(n);
            Hessian_t.resize(n,n);
            s.resize(n);
            y.resize(n);
            Nabla_F.resize(n);
            Prev_Nabla_F.resize(n);
            
        }
        
        inline Scalar DirectionRate(int k) const {return rate;}
        
        inline int RandomGenerator(int N) const
        {
            //std::uniform_int_distribution<int> distribution(0,param.N - 1);
            //return distribution(random_generator);
            return rand() % N;
        }
        
        inline void VectorGenerator(std::vector<Scalar> origin, Vector& Vec) const
        {
            int length = origin.size();
            Vec.resize(length);
            for (int i = 0; i <length ;i++)
                Vec(i) = origin[i];
        }
        
        /////////////////////////////////////////////
        ////// tool function/////////////////////////
        inline void UpdateHessian(Matrix& H_t, int t)
        {
            int M = param.M;
            int m_prime = std::min(M,t);
            long size = H_t.rows();
            Matrix H(size,size);                                            //square Hessian matrix
            Matrix frac1 = History_s[t-1].transpose() * History_y[t-1];
            Matrix frac2 = History_y[t-1].transpose() * History_y[t-1];
            Matrix I(size,size);
            I.setIdentity();
            H = frac1(0,0) / frac2(0,0) * I;
            for (int j = t - m_prime; j < t; j++  )
            {
                Vector s_j = History_s[j];
                Vector y_j = History_y[j];
                Scalar pho = 1/ ((y_j.transpose() * s_j)(0,0));
                H = (I - pho * s_j * y_j.transpose()) * H * (I - pho * y_j * s_j.transpose()) + pho * s_j * s_j.transpose();
            }
            H_t = H;
        }
        
        
        template <typename Func>
        inline Scalar F_Evaluation(Func& f, Vector Omega) const
        {
            int N = param.N;
            Scalar result = Scalar(0);
            for (int index=0;index<N;index++)
            {
                Vector vec_x;
                Vector vec_z;
                VectorGenerator(param.input_data[index],vec_x);
                vec_z.resize(1);
                vec_z(0) = param.output_data[index];
                result += f.Value(vec_x,vec_z,Omega);
            }
            return result;
        }
        
        
        template <typename Func>
        inline Vector StochasticGrad(Func& f, Vector& Omega)
        {
            int n  = Omega.size();
            int N  = param.N;    //size of dataset
            int b  = param.b;    //size of a batch;
            Vector result = Vector::Zero(n);
            for (int i=0;i<b;i++)
            {
                int index = RandomGenerator(N);
                Vector vec_x;
                Vector vec_z;
                VectorGenerator(param.input_data[index],vec_x);
                vec_z.resize(1);
                vec_z(0) = param.output_data[index];
                std::cout<<"a sample vector x is : "<<vec_x.transpose()<<std::endl;
                std::cout<<"a sample output z is : "<<vec_z.transpose()<<std::endl;
                std::cout<<"single value of a gradient : "<<(f.Gradient(vec_x,vec_z,Omega)).transpose();
                std::cout<<"\n\n";
                result += f.Gradient(vec_x,vec_z,Omega);
            }
            result = (Scalar(1) / Scalar(b)) * result;
            std::cout<<"Stochastic Grad Result : " << result.transpose()<<std::endl;
            return result;
        }
        
        
        
        template <typename Func>
        inline Matrix StochasticHessian(Func& f, Vector& Omega_bar)
        {
            int n = Omega_bar.size();
            int N = param.N;
            int b_H = param.b_H;
            Matrix result = Matrix::Zero(n,n);
            for (int i=0;i<b_H;i++)
            {
                int index = RandomGenerator(N);
                Vector vec_x;
                Vector vec_z;
                VectorGenerator(param.input_data[index],vec_x);
                vec_z.resize(1);
                vec_z(0) = param.output_data[index];
                result += f.Hessian(vec_x,vec_z,Omega_bar);
            }
            result = (Scalar(1) / Scalar(b_H)) * result;
            return result;
        }
        
        template <typename Func>
        inline Vector Special_StochaticHessian_S(Func& f, Vector& Omega_bar, Vector& S)
        {
            int n = Omega_bar.size();
            int N = param.N;
            int b_H = param.b_H;
            Vector result = Vector::Zero(n);
            for (int i=0;i<b_H;i++)
            {
                int index = RandomGenerator(N);
                Vector vec_x;
                Vector vec_z;
                VectorGenerator(param.input_data[index],vec_x);
                vec_z.resize(1);
                vec_z(0) = param.output_data[index];
                result += f.Hessian_s(vec_x,vec_z,Omega_bar,S);
            }
            result = (Scalar(1) / Scalar(b_H)) * result;
            return result;
        }

    public:
        SQNsolver(const SQNpp<Scalar>& param, SQNreport<Scalar>& report) :
        param(param), Report(report)
        {
            param.check_param();
        }
        
        template <typename Func>
        inline void minimizer(Func& f, Vector& Omega, Scalar& fx){
                                                                                        //initillization
            int n = param.n;                                                            //dimension
            reset(n);
            int t = -1;                                                                 //pointer
            int k = 1;                                                                  //counter
            int L = param.L;                                                            //window size
            Scalar epsilon = param.epsilon;                                             //tolerance
            omega_bar = Vector::Zero(n);                                                //set to zero
            omega = Omega;                                                    //set to zero
            int reportCounter = 0;
            
            Report.StartTiming();
            for (;;)
            {
                if (reportCounter<10) Report.StartTiming(); //++
                Nabla_F = StochasticGrad(f,omega);                                      //batch gradient
                if (reportCounter<10) Report.EndTiming(std::to_string(reportCounter)+"  Stochastic Grad"); //++
                omega_bar += omega;                                                     //accmulate gradient direction
                std::cout<<"\n\n";
                std::cout<<"Omega Shows: "<<omega.transpose()<<std::endl;
                std::cout<<"\n\n";
                if (k <= 2*L)                                                           //two different updading methods
                    omega = omega - DirectionRate(k) * Nabla_F;
                else{
                    if (reportCounter<10+2*L) Report.StartTiming();
                    omega = omega - DirectionRate(k) * Hessian_t * Nabla_F;
                    if (reportCounter<10+2*L) Report.EndTiming(std::to_string(reportCounter)+" Omega updating with Hessian");

                }
                if (k % L == 0)
                {
                    t++;
                    omega_bar = omega_bar / Scalar(L);                                  //get the very direction which is really
                                                                                        //used for computing hessian
                    if (t > 0 )
                    {
                        if (reportCounter<10) Report.StartTiming(); //++
                        
                        s = omega_bar - prev_omega_bar;                                 //s in standard L-BFGS method
                        y = Special_StochaticHessian_S(f,omega_bar,s);                  //y in standard L_BFGS method
                        if (reportCounter<10) Report.EndTiming(std::to_string(reportCounter) + "  Updating Nabla_Square_F and y"); //++
                        History_s.push_back(s);                                         //record all the s
                        History_y.push_back(y);                                         //record all the y
                        if (reportCounter<10) Report.StartTiming(); //++
                        UpdateHessian(Hessian_t,t);                                     //using new vector to update Hessian
                        if (reportCounter<10) Report.EndTiming(std::to_string(reportCounter)+"  Updating Hessian"); //++
                    }
                    prev_omega_bar = omega_bar;
                    omega_bar = Vector::Zero(n);
                }
                k++;                                                                    //increment of counter
                                                                                        //test whether getting convergence now
                fx = F_Evaluation(f, omega);                                            //update new fx value
                Scalar gnorm = (Nabla_F).norm();                                        //update new gradient norm
                Report.RecordValue(fx);
                Report.RecordGradient(gnorm);
                if (gnorm < epsilon) {                                                  //the gradient will not change
                    //do some report
                    Omega = prev_omega_bar;                                             //get the final optimal parameter Omega
                    break;                                                              //exit the loop
                }
                                                                                        //else keep the old gradient
                //Prev_Nabla_F = Nabla_F;                                               //in order happen some wired gradient
                                                                                        //function, this is for backup plan.
                reportCounter++;
            }
            Report.EndTiming("Main Loop ends"); //++
            Report.WriteLog(k,n); //++
        }
        
        
        };
    
}

#endif /* SQN_h */
