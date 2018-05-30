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
#include <math.h>

namespace SQNpp {
    template <typename Scalar>
    class SQNsolver{
    private:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::Map<Vector> MapVec;
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
        Matrix History_s;
        Matrix History_y;
        SQNreport<Scalar>   Report;    //for generating report
        int end;         //pointer
        
        inline void reset(int n)
        {
            const int m = param.m;
            omega.resize(n);
            omega_bar.resize(n);
            prev_omega_bar.resize(n);
            Hessian_t.resize(n,n);
            s.resize(n);
            y.resize(n);
            History_s.resize(n,m);
            History_y.resize(n,m);
            Nabla_F.resize(n);
            Prev_Nabla_F.resize(n);
            end  = 0;
        }
        
        inline Scalar DirectionRate(int k) const {return param.alpha / Scalar(k) + Scalar(0.1);}
        
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
        
        inline void UpdateHessian(Matrix& H_t, int t,Vector s,Vector y)
        {
            const int n = s.size();
            MapVec svec(&History_s(0,end),n);
            MapVec yvec(&History_y(0,end),n);
            svec.noalias() = s;
            yvec.noalias() = y;
            Matrix H(n,n);                                            //square Hessian matrix
            Scalar frac1 = s.dot(y);
            Scalar frac2 = y.dot(y);
            Matrix I(n,n);
            I.setIdentity();
            H = frac1 / frac2 * I;
            int bound = std::min(param.m,t);
            end = (end+1) % param.m;
            int j = end;
            for (int i = 0; i<bound; i++  )
            {
                j = (j + param.m - 1) % param.m;
                MapVec s_j(&History_s(0,j),n);
                MapVec y_j(&History_y(0,j),n);
                Scalar pho = 1/ y_j.dot(s_j);
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
            result = result / Scalar(N);
            assert(std::isnan(result) == false);
            return result;
        }
        
        inline Scalar Entropy(Eigen::VectorXd x, Eigen::VectorXd w)
        {
            double xTw = x.dot(w);
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
                result += f.Gradient(vec_x,vec_z,Omega);
            }
            result = (Scalar(1) / Scalar(b)) * result;
            assert(std::isnan(result.norm()) == false);
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
            assert(std::isnan(result.norm())==false);
            return result;
        }
        
        inline void AdamMax(Vector & omega, Vector & Grad, int t){
            static Scalar alpha = 0.002;
            static Scalar beta1 = 0.9;
            static Scalar beta2 = 0.999;
            static Vector m_t = Vector::Zero(omega.size());
            static Scalar u_t = 0;
            m_t = beta1 * m_t + (1 - beta1) * Grad;
            u_t = std::max(beta2*u_t,std::sqrt(Grad.norm()));
            Scalar rate = (alpha / (1 - Scalar(std::pow(beta1,Scalar(t))))) / u_t;
            omega = omega - rate * m_t;
            
        }
        
        inline void Adam(Vector & omega, Vector & Grad, int t){
            static Scalar alpha = 0.001;
            static Scalar beta1 = 0.7;
            static Scalar beta2 = 0.8;
            static Vector m_t = Vector::Zero(omega.size());
            static Scalar v_t = 0;
            static double epsilon = 10e-8;
            m_t = beta1 * m_t + (1 - beta1) * Grad;
            v_t = beta2*v_t + (1 - beta2) * Grad.dot(Grad);
            Vector m_t_hat = (Scalar(1.0) / (1.0 - Scalar(std::pow(beta1,Scalar(t))))) * m_t;
            Scalar v_t_hat = (Scalar(1.0) / (1.0 - Scalar(std::pow(beta2,Scalar(t))))) * v_t;
            omega = omega - alpha / (std::sqrt(v_t_hat) + epsilon) * m_t_hat;
        }
        
        inline void SGD(Vector & omega, Vector & Grad, int t){
            omega = omega - (param.alpha / Scalar(t) + Scalar(0.001)) * Grad;
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
            Scalar prev_gnorm = 0;                                                      //another exit method
            omega_bar = Vector::Zero(n);                                                //set to zero
            omega = Omega;                                                    //set to zero
            
            clock_t startTime,endTime;
            for (;;)
            {
                //if (k<20) Report.StartTiming(); //++
                if (k < 20) startTime = clock();
                Nabla_F = StochasticGrad(f,omega);                                      //batch gradient
                //if (k<20) Report.EndTiming(std::to_string(k)+"  Stochastic Grad"); //++
                if (k < 20) {
                    endTime = clock();
                    double time = (double)(endTime - startTime) / CLOCKS_PER_SEC;
                    Report.Durations["Stochastic Grad"].push_back(time);
                }
                omega_bar += omega;                                                     //accmulate gradient direction
                Vector Grad;
                if (k <= 2*L)                                                           //two different updading methods
                    Grad =  Nabla_F;
                else
                    Grad = Hessian_t * Nabla_F;
                
                AdamMax(omega, Grad, k);
                
                if (k % L == 0)
                {
                    t++;
                    omega_bar = omega_bar / Scalar(L);                                  //get the very direction which is really
                                                                                        //used for computing hessian
                    if (t > 0 )
                    {
                        if (t<20) startTime = clock();
                        s = omega_bar - prev_omega_bar;                                 //s in standard L-BFGS method
                        y = Special_StochaticHessian_S(f,omega_bar,s);                  //y in standard L_BFGS method
                        if (t<20) {
                            endTime = clock();
                            double time = (double)(endTime - startTime) / CLOCKS_PER_SEC;
                            Report.Durations["Nabla_F"].push_back(time);
                        }
                        if (t<20) startTime = clock();                        UpdateHessian(Hessian_t,t,s,y);                                     //using new vector to update Hessian
                        if (t<20) {
                            endTime = clock();
                            double time = (double)(endTime - startTime) / CLOCKS_PER_SEC;
                            Report.Durations["Hessian Updadting"].push_back(time);
                        }
                    }
                    prev_omega_bar = omega_bar;
                    omega_bar = Vector::Zero(n);
                }
                k++;                                                                    //increment of counter
                                                                                        //test whether getting convergence now
                fx = F_Evaluation(f, omega);                                            //update new fx value
                Scalar gnorm = (Nabla_F).norm();                                        //update new gradient norm
                if (k % 2 ==0){
                    std::cout<<"\n\n";
                    std::cout<<"iterations : "<<k<< "  ;  "<<"fx value = "<<fx << std::endl;
                    std::cout<<"iterations : "<<k<< "  ;  "<<"gradient norm = "<<gnorm << std::endl;

                }
                
                if (gnorm < epsilon) {                                                  //the gradient will not change
                    //do some report
                    Omega = prev_omega_bar;                                             //get the final optimal parameter Omega
                    break;                                                              //exit the loop
                }
                if (k > param.max_iterations)
                    break;
                
                if (std::abs(prev_gnorm - gnorm)< epsilon)
                    break;
                //else keep the old gradient
                //Prev_Nabla_F = Nabla_F;                                               //in order happen some wired gradient
                                                                                        //function, this is for backup plan.
                prev_gnorm = gnorm;
            }
            Report.WriteLog(k,n,t); //++
        }
        
        template <typename Func>
        inline void minimizer_enhanced(Func& f, Vector& Omega, Scalar& fx,Scalar eta = Scalar(0.2), Scalar tau = Scalar(1.5)){
                                                                                        //initillization
            int n = param.n;                                                            //dimension
            reset(n);
            int t = -1;                                                                 //pointer
            int k = 1;                                                                  //counter
            int j = 0;                                                                  //counter to help update window size
            int L = param.L;                                                            //window size
            int L_limit = param.L;                                                      //maximal size of L
            int localcounter = 0;                                                       //local counter for exit, using a
                                                                                        //function idicator
            Scalar prev_gnorm = 0;
            Scalar epsilon = param.epsilon;                                             //tolerance
            omega_bar = Vector::Zero(n);                                                //set to zero
            omega = Omega;                                                              //set to zero
            Vector Drt;                                                                 //descent direction
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
                omega_bar += omega;                                                      //accmulate gradient direction
                Scalar alpha_k = DirectionRate(k);
                if (k <= 2*L_limit)                                                           //two different updading methods
                {
                    Drt = Nabla_F;
                    omega = omega -  alpha_k * Drt;
                }
                else{
                    if (k<10+2*L_limit) Report.StartTiming();
                    Drt = Hessian_t * Nabla_F;
                    omega = omega - alpha_k *  Drt;
                    if (k<10+2*L_limit) Report.EndTiming(std::to_string(k)+" Omega updating with Hessian");

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
                        if (t<10) Report.StartTiming(); //++
                        UpdateHessian(Hessian_t,t,s,y);                                     //using new vector to update
                        if (t<10) Report.EndTiming(std::to_string(t)+"  Updating Hessian"); //++
                    }
                    //update the size of window
                    fx = F_Evaluation(f, omega_bar);                                            //update new fx value
                    if (t>0)
                    {
                        Vector Grad = StochasticGrad(f,prev_omega_bar);
                        Scalar gTp =  Grad.dot(Grad);    //should be less than zero
                        if ((prev_fx - fx) >=  eta * gTp)
                        {
                            L = int(tau * Scalar(L) ) + 1;
                            L = std::min(L,L_limit);
                        }
                        else
                        {
                            L = int(Scalar(L) / tau) - 1;
                            L = std::max(L,1);
                        }
                        std::cout<<"iterations : "<<k<< "  ;  "<<"gTp = "<<gTp<<std::endl;
                        std::cout<<"iterations : "<<k<< "  ;  "<<"fx descent = "<<(prev_fx - fx) << std::endl;
                        if (prev_fx - fx < epsilon)
                            localcounter ++;
                        if (localcounter > 5)
                            break;
                    }
                    prev_omega_bar = omega_bar;
                    omega_bar = Vector::Zero(n);
                    prev_fx = fx;
                    Report.AddRecordValue("L",L);
                }
                k++;                                                                    //increment of counter
                fx = F_Evaluation(f, omega);                                                               //test whether getting convergence now
                Scalar gnorm = (Nabla_F).norm();                                        //update new gradient norm
                if (k % 2 ==0){
                    std::cout<<"\n\n";
                    std::cout<<"iterations : "<<k<< "  ;  "<<"fx value = "<<fx << std::endl;
                    std::cout<<"iterations : "<<k<< "  ;  "<<"gradient norm = "<<gnorm << std::endl;
                    std::cout<<"iterations : "<<k<< "  ;  "<<"window size = "<<L << std::endl;
                    
                }

                Report.AddRecordValue("g_norm",gnorm);                                      //for record
                Report.AddRecordValue("fx",fx);
                Report.AddRecordVector("Omega",omega);
                Report.AddRecordVector("Nabla_F",Nabla_F);

                
                if (gnorm < epsilon) {                                                  //the gradient will not change
                    //do some report
                    Omega = prev_omega_bar;                                             //get the final optimal parameter Omega
                    break;                                                              //exit the loop
                }
                if (k > param.max_iterations)
                    break;
                
                if (std::abs(prev_gnorm - gnorm)< epsilon)
                    break;
                
                                                                                        //else keep the old gradient
                //Prev_Nabla_F = Nabla_F;                                               //in order happen some wired gradient
                                                                                        //function, this is for backup plan.
                prev_gnorm = gnorm;
            }
            Report.EndTiming("Main Loop ends"); //++
            Report.WriteLog(k,n,t); //++
        }

        };
    
}

#endif /* SQN_h */
