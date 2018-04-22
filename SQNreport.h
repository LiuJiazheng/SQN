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
        
        std::vector<uint64_t> start;            //start counting time
        uint64_t end;              //end time
        std::map<std::string,double> TimeSeries;        //different part time record
        std::vector<Scalar> iteration_value;
        std::vector<Scalar> iteration_gradient;
        int ptr;
        const SQNpp<Scalar>&   param;
        
    public:
        SQNreport(const SQNpp<Scalar>&   param) : param(param) {ptr = 0; }
        
        void StartTiming()  {start.push_back(rdtsc());ptr++;}
        void EndTiming(std::string str)
        {
            end = rdtsc();
            ptr--;
            double DurationTime = (double)(end - start[ptr]) / CLOCKS_PER_SEC ;   //get the secs
            TimeSeries.insert(std::pair<std::string,double>(str,DurationTime));
        }
        void RecordValue(Scalar Value) {iteration_value.push_back(Value);}
        void RecordGradient(Scalar GradNorm) {iteration_gradient.push_back(GradNorm);}
        
        void WriteLog(int k,int n)
        {
            //write log
            std::ofstream LogFile ("Log.txt");
            if (LogFile.is_open())
            {
                LogFile << "Memeory Limit (iteration times per updation) : "<< param.M <<"\n";
                LogFile << "Window Size : " << param.L << "\n";
                LogFile << "Gradient Batch Size: " << param.b <<"\n";
                LogFile << "Hessian Batch Size: " << param.b_H <<"\n";
                LogFile << "Demension : " << n << "\n";
                LogFile << "Iteration Times : " << k << "\n";
                LogFile << "\n";
                LogFile << "<<<<<<<<<<<<<<<<<<<<Detials>>>>>>>>>>>>>>>>>>>>>" << "\n";
                LogFile << "\n";
                for (std::map<std::string,double>::iterator it = TimeSeries.begin();
                     it != TimeSeries.end();
                     ++it )
                {
                    LogFile << it->first << ":\n";
                    LogFile << it->second  << "  "<< "secs \n";
                    LogFile << "\n";
                }
                LogFile.close();
            }
            
            //write value data
            std::ofstream ValueData("Value.txt");
            if (ValueData.is_open())
            {
                for ( typename std::vector<Scalar>::iterator it = iteration_value.begin();
                     it != iteration_value.end();
                     ++it)
                    ValueData << *it <<"    ";
                ValueData.close();
            }
            
            //write gradient data
            std::ofstream GradData("Grad.txt");
            if (ValueData.is_open())
            {
                for ( typename std::vector<Scalar>::iterator it = iteration_gradient.begin();
                     it != iteration_gradient.end();
                     ++it)
                    GradData << *it <<"    ";
                ValueData.close();
            }

        }
        
    };
}

#endif /* SQNreport_h */
