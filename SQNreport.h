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
        std::map<std::string, std::vector<Scalar>> RecordValue;
        std::map<std::string, std::vector<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>> RecordVector;
        int ptr;
        const SQNpp<Scalar>&   param;
        std::string FilePath;
        
    public:
        SQNreport(const SQNpp<Scalar>&   Param, const std::string path)  : 
        param(Param), FilePath(path)
        {
            
            ptr = 0; 
        }
        
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
                    LogFile << (it->second).count()  << "  "<< "mili secs \n";
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
                        Data<< *itr << "\n";
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
                        Data<< (*itr).transpose() << "\n";
                    Data.close();
                }
            }

        }
        
    };
}

#endif /* SQNreport_h */
