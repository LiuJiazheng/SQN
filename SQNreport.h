//
//  SQNreport.h
//  SQN
//
//  Created by Jiazheng LIU on 4/20/18.
//  Copyright © 2018 Jiazheng LIU. All rights reserved.
//

#ifndef SQNreport_h
#define SQNreport_h
#include "SQNparam.h"
#include <map>
#include <string>
#include <fstream>
#include <cstdlib>
#include <stdint.h>
#include <chrono>

//  Windows
#ifdef _WIN32

#include <intrin.h>
uint64_t rdtsc(){
    return __rdtsc();
}

//  Linux/GCC
#else

uint64_t rdtsc(){
    unsigned int lo,hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
}

#endif

namespace SQNpp{
    template <typename Scalar>
    class SQNreport
    {
    private:
        
        std::vector<std::chrono::steady_clock::time_point> start;            //start counting time
        uint64_t end;              //end time
        std::map<std::string,std::chrono::duration<long, std::milli>> TimeSeries;        //different part time record
        std::vector<Scalar> iteration_value;
        std::vector<Scalar> iteration_gradient;
        int ptr;
        const SQNpp<Scalar>&   param;
        
    public:
        SQNreport(const SQNpp<Scalar>&   param) : param(param) {ptr = 0; }
        
        void StartTiming()  {
            auto begin = std::chrono::high_resolution_clock::now();
            start.push_back(begin);
            ptr++;
        }
        void EndTiming(std::string str)
        {
            auto end = std::chrono::high_resolution_clock::now();
            ptr--;
            std::chrono::duration<long, std::milli> DurationTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start[ptr])  ;   //get the mi secs
            TimeSeries.insert(std::pair<std::string,std::chrono::duration<long, std::milli>> (str,DurationTime));
        }
        void RecordValue(Scalar Value) {iteration_value.push_back(Value);}
        void RecordGradient(Scalar GradNorm) {iteration_gradient.push_back(GradNorm);}
        
        void WriteLog(int k,int n,int t)
        {
            //write log
            std::ofstream LogFile ("/Users/LiuJiazheng/Documents/Optimazation/Data/Out/Log.txt");
            if (LogFile.is_open())
            {
                LogFile << "Memeory Limit (iteration times per updation) : "<< param.M <<"\n";
                LogFile << "Window Size : " << param.L << "\n";
                LogFile << "Gradient Batch Size: " << param.b <<"\n";
                LogFile << "Hessian Batch Size: " << param.b_H <<"\n";
                LogFile << "Demension : " << n << "\n";
                LogFile << "Iteration Times : " << k << "\n";
                LogFile << "\n";
                LogFile << "Hessian Updating Times : " << t << "\n";
                LogFile << "<<<<<<<<<<<<<<<<<<<<Detials>>>>>>>>>>>>>>>>>>>>>" << "\n";
                LogFile << "\n";
                for (auto it = TimeSeries.begin();
                     it != TimeSeries.end();
                     ++it )
                {
                    LogFile << it->first << ":\n";
                    LogFile << (it->second).count()  << "  "<< "msecs \n";
                    LogFile << "\n";
                }
                LogFile.close();
            }
            
            //write value data
            std::ofstream ValueData("/Users/LiuJiazheng/Documents/Optimazation/Data/Out/Value.txt");
            if (ValueData.is_open())
            {
                for ( typename std::vector<Scalar>::iterator it = iteration_value.begin();
                     it != iteration_value.end();
                     ++it)
                    ValueData << *it <<"    ";
                ValueData.close();
            }
            
            //write gradient data
            std::ofstream GradData("/Users/LiuJiazheng/Documents/Optimazation/Data/Out/Grad.txt");
            if (ValueData.is_open())
            {
                for ( typename std::vector<Scalar>::iterator it = iteration_gradient.begin();
                     it != iteration_gradient.end();
                     ++it)
                    GradData << *it <<"    ";
                GradData.close();
            }

        }
        
    };
}

#endif /* SQNreport_h */
