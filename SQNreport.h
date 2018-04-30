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
#include <time.h>

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
        
        std::vector<clock_t>  start;            //start counting time
        clock_t end;              //end time
        std::map<std::string,double> TimeSeries;        //different part time record
        std::map<std::string, std::vector<Scalar>> RecordValue;
        std::map<std::string, std::vector<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>> RecordVector;
        int ptr;
        const SQNheader<Scalar>&   param;
        std::string FilePath;
        
    public:
        SQNreport(const SQNheader<Scalar>&   Param, const std::string path)  : 
        param(Param), FilePath(path)
        {
            
            ptr = 0; 
        }
        
        void StartTiming()  {
            clock_t begin = clock();
            start.push_back(begin);
            ptr++;
        }
        void EndTiming(std::string str)
        {
            clock_t end = clock();
            ptr--;
            double duration = (double)(end - start[ptr])/CLOCKS_PER_SEC;
            TimeSeries.insert(std::pair<std::string,double> (str,duration));
        }
        
        void StartRecordValue(std::string VariableName)
        {
            std::vector<Scalar> v_record;
            RecordValue.emplace(std::make_pair(VariableName,v_record));
        }
        
        
        void AddRecordValue(std::string VariableName,Scalar Value)
        {
            RecordValue[VariableName].push_back(Value);
        }
        
        void StartRecordVector(std::string VariableName)
        {
            std::vector<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> v_record;
            RecordVector.emplace(std::make_pair(VariableName,v_record));
        }
        
        void AddRecordVector(std::string VariableName, Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector)
        {
            RecordVector[VariableName].push_back(Vector);
        }
        
        
        
        void WriteLog(int k,int n,int t)
        {
            //write log
            //FilePath = "/Users/LiuJiazheng/Documents/Optimazation/Data/Out/";
            std::ofstream LogFile (FilePath + "Log.txt");
            if (LogFile.is_open())
            {
                LogFile << "Memeory Limit (iteration times per updation) : "<< param.m <<"\n";
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
                    LogFile << (it->second)  << "  "<< "micro secs \n";
                    LogFile << "\n";
                }
                LogFile.close();
            }
            
            //write value data
            for (auto it = RecordValue.begin(); it != RecordValue.end(); ++it)
            {
                std::string FileName = it -> first;
                std::ofstream Data(FilePath + FileName + ".txt");
                if (Data.is_open())
                {
                    auto vector = it ->second;
                    for (auto itr = vector.begin(); itr != vector.end();++itr)
                        Data<< "\n"<< *itr << "\n\n";
                    Data.close();
                }
            }
            
            //write vector data
            for (auto it = RecordVector.begin(); it != RecordVector.end(); ++it)
            {
                std::string FileName = it -> first;
                std::ofstream Data(FilePath + FileName + ".txt");
                if (Data.is_open())
                {
                    auto vector = it ->second;
                    for (auto itr = vector.begin(); itr != vector.end();++itr)
                        Data<< "\n" << (*itr).transpose() << "\n\n";
                    Data.close();
                }
            }

        }
        
    };
}

#endif /* SQNreport_h */
