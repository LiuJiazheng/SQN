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
        
        const SQNheader<Scalar>&   param;  // Parameters to control the SQN algorithm
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
            omega.resize(n);
            omega_bar.resize(n);
            prev_omega_bar.resize(n);
            Hessian_t.resize(n,n);
            s.resize(n);
            y.resize(n);
            Nabla_F.resize(n);
            Prev_Nabla_F.resize(n);
            
        }
        
        inline Scalar DirectionRate(int k) const {return param.alpha / Scalar(k);}
        
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
        
        //When iteration researches the maximal memory limit M, we need to erase some space who keeps the history of vector
        //s and vector y
        inline void Erase() 
        {
            int Size = History_s.size();
            if (Size > param.M)
            {
                int length  = Size - param.M;
                History_s.erase(History_s.begin(),History_s.begin()+length);
                History_y.erase(History_y.begin(),History_y.begin()+length);
            }
        }
        
        
        
        inline void UpdateHessian(Matrix& H_t, int t)
        {
            int M = param.M;
            int m_prime = std::min(M,t);
            if (m_prime == M)
                Erase();
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
            return result / Scalar(N);
        }
        
        inline Scalar Entropy(Eigen::VectorXd x, Eigen::VectorXd w)
        {
            double xTw = (x.transpose() * w)(0,0);
            return  1.0 / (1.0 + exp(-1.0 * xTw));
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
                //std::cout<<"a sample vector x is : "<<vec_x.transpose()<<std::endl;
                //std::cout<<"a sample output z is : "<<vec_z.transpose()<<std::endl;
                //std::cout<<"now omega is : "<<Omega.transpose() <<std::endl;
                //std::cout<<"a sample xTw is : " << vec_x.transpose() * Omega << std::endl;
                //std::cout<<"a sample entropy is : " << Entropy(vec_x, Omega)<<std::endl;
                //std::cout<<"single value of a gradient : "<<(f.Gradient(vec_x,vec_z,Omega)).transpose();
                //std::cout<<"\n\n";
                result += f.Gradient(vec_x,vec_z,Omega);
            }
            result = (Scalar(1) / Scalar(b)) * result;
            //std::cout<<"Stochastic Grad Result : " << result.transpose()<<std::endl;
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
        SQNsolver(const SQNheader<Scalar>& param, SQNreport<Scalar>& report) :
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
            Report.StartTiming();
            //define the variable you want to trace
            Report.StartRecordValue("g_norm");
            Report.StartRecordValue("fx");
            Report.StartRecordVector("Omega");
            Report.StartRecordVector("Nabla_F");
            for (;;)
            {
                if (k<10) Report.StartTiming(); //++
                Nabla_F = StochasticGrad(f,omega);                                      //batch gradient
                if (k<10) Report.EndTiming(std::to_string(k)+"  Stochastic Grad"); //++
                omega_bar += omega;                                                     //accmulate gradient direction
                if (k <= 2*L)                                                           //two different updading methods
                    omega = omega - 0.5 * DirectionRate(k) * Nabla_F;
                else{
                    if (k<10+2*L) Report.StartTiming();
                    omega = omega - DirectionRate(k) * Hessian_t * Nabla_F;
                    if (k<10+2*L) Report.EndTiming(std::to_string(k)+" Omega updating with Hessian");

                }
                if (k % L == 0)
                {
                    t++;
                    omega_bar = omega_bar / Scalar(L);                                  //get the very direction which is really
                                                                                        //used for computing hessian
                    if (t > 0 )
                    {
                        if (t<10) Report.StartTiming(); //++
                        
                        s = omega_bar - prev_omega_bar;                                 //s in standard L-BFGS method
                        y = Special_StochaticHessian_S(f,omega_bar,s);                  //y in standard L_BFGS method
                        if (t<10) Report.EndTiming(std::to_string(t) + "  Updating Nabla_Square_F and y"); //++
                        History_s.push_back(s);                                         //record all the s
                        History_y.push_back(y);                                         //record all the y
                        if (t<10) Report.StartTiming(); //++
                        UpdateHessian(Hessian_t,t);                                     //using new vector to update Hessian
                        if (t<10) Report.EndTiming(std::to_string(t)+"  Updating Hessian"); //++
                    }
                    prev_omega_bar = omega_bar;
                    omega_bar = Vector::Zero(n);
                }
                k++;                                                                    //increment of counter
                                                                                        //test whether getting convergence now
                fx = F_Evaluation(f, omega);                                            //update new fx value
                if (k % 10 ==0){
                    std::cout<<"\n";
                    std::cout<<"Nabla Shows : " << Nabla_F.transpose() << std::endl;
                    std::cout<<"\n\n";
                    std::cout<<"Omega Shows : "<<omega.transpose()<<std::endl;
                    std::cout<<"\n\n";
                    std::cout<<"fx value : "<<fx << std::endl;
                }
                Scalar gnorm = (Nabla_F).norm();                                        //update new gradient norm
                
            Report.AddRecordValue("g_norm",gnorm);                                      //for record
                Report.AddRecordValue("fx",fx);
                Report.AddRecordVector("Omega",omega);
                Report.AddRecordVector("Nabla_F",Nabla_F);
                
                if (gnorm < epsilon) {                                                  //the gradient will not change
                    //do some report
                    Omega = prev_omega_bar;                                             //get the final optimal parameter Omega
                    break;                                                              //exit the loop
                }
                                                                                        //else keep the old gradient
                //Prev_Nabla_F = Nabla_F;                                               //in order happen some wired gradient
                                                                                        //function, this is for backup plan.
            }
            Report.EndTiming("Main Loop ends"); //++
            Report.WriteLog(k,n,t); //++
        }
        
        template <typename Func>
        inline void minimizer_enhanced(Func& f, Vector& Omega, Scalar& fx,Scalar eta = Scalar(0.7), Scalar tau = Scalar(1.5)){
                                                                                        //initillization
            int n = param.n;                                                            //dimension
            reset(n);
            int t = -1;                                                                 //pointer
            int k = 1;                                                                  //counter
            int j = 0;                                                                  //counter to help update window size
            int L = param.L;                                                            //window size
            int L_limit = param.L;                                                      //maximal size of L
            Scalar epsilon = param.epsilon;                                             //tolerance
            omega_bar = Vector::Zero(n);                                                //set to zero
            omega = Omega;                                                    //set to zero
            Scalar prev_fx = Scalar(0);                                         
            Report.StartTiming();
            //define the variable you want to trace
            Report.StartRecordValue("g_norm");
            Report.StartRecordValue("fx");
            Report.StartRecordVector("Omega");
            Report.StartRecordVector("Nabla_F");
            Report.StartRecordValue("L");
            for (;;)
            {
                if (k<10) Report.StartTiming(); //++
                Nabla_F = StochasticGrad(f,omega);                                      //batch gradient
                if (k<10) Report.EndTiming(std::to_string(k)+"  Stochastic Grad"); //++
                omega_bar += omega;                                                     //accmulate gradient direction
                if (k <= 2*L)                                                           //two different updading methods
                    omega = omega - 0.5 * DirectionRate(k) * Nabla_F;
                else{
                    if (k<10+2*L) Report.StartTiming();
                    omega = omega - DirectionRate(k) * Hessian_t * Nabla_F;
                    if (k<10+2*L) Report.EndTiming(std::to_string(k)+" Omega updating with Hessian");

                }
                j++;
                if (j == L)
                {
                    j = 0;
                    t++;
                    omega_bar = omega_bar / Scalar(L);                                  //get the very direction which is really
                                                                                        //used for computing hessian
                    if (t > 0 )
                    {
                        if (t<10) Report.StartTiming(); //++
                        
                        s = omega_bar - prev_omega_bar;                                 //s in standard L-BFGS method
                        y = Special_StochaticHessian_S(f,omega_bar,s);                  //y in standard L_BFGS method
                        if (t<10) Report.EndTiming(std::to_string(t) + "  Updating Nabla_Square_F and y"); //++
                        History_s.push_back(s);                                         //record all the s
                        History_y.push_back(y);                                         //record all the y
                        if (t<10) Report.StartTiming(); //++
                        UpdateHessian(Hessian_t,t);                                     //using new vector to update Hessian
                        if (t<10) Report.EndTiming(std::to_string(t)+"  Updating Hessian"); //++
                    }
                    //update the size of window
                    fx = F_Evaluation(f, omega_bar);                                            //update new fx value
                    if (t>0)
                    {
                        Scalar gTp =  Nabla_F.dot(Nabla_F);    //should be less than zero
                        if ((prev_fx - fx) > eta * gTp)
                        {
                            L = int(tau * Scalar(L) ) + 1;
                            L = std::min(L,L_limit);
                        }
                        else
                        {
                            L = int(Scalar(L) / tau) - 1;
                            L = std::max(L,1);
                        }
                    }
                    prev_omega_bar = omega_bar;
                    omega_bar = Vector::Zero(n);
                    prev_fx = fx;
                    Report.AddRecordValue("L",L);
                }
                k++;                                                                    //increment of counter
                fx = F_Evaluation(f, omega);                                                               //test whether getting convergence now
                if (k % 10 ==0){
                    std::cout<<"\n";
                    std::cout<<"Nabla Shows : " << Nabla_F.transpose() << std::endl;
                    std::cout<<"\n\n";
                    std::cout<<"Omega Shows : "<<omega.transpose()<<std::endl;
                    std::cout<<"\n\n";
                    std::cout<<"fx value : "<<fx << std::endl;
                }
                Scalar gnorm = (Nabla_F).norm();                                        //update new gradient norm
                
                Report.AddRecordValue("g_norm",gnorm);                                      //for record
                Report.AddRecordValue("fx",fx);
                Report.AddRecordVector("Omega",omega);
                Report.AddRecordVector("Nabla_F",Nabla_F);

                
                if (gnorm < epsilon) {                                                  //the gradient will not change
                    //do some report
                    Omega = prev_omega_bar;                                             //get the final optimal parameter Omega
                    break;                                                              //exit the loop
                }
                                                                                        //else keep the old gradient
                //Prev_Nabla_F = Nabla_F;                                               //in order happen some wired gradient
                                                                                        //function, this is for backup plan.
            }
            Report.EndTiming("Main Loop ends"); //++
            Report.WriteLog(k,n,t); //++
        }

        };
    
}

#endif /* SQN_h */
