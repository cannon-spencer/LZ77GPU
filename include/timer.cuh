//
// Created by yu hong on 11/17/24.
//

#ifndef LIBCUBWT_TIMER_CUH
#define LIBCUBWT_TIMER_CUH

#include <chrono>
#include <string>
#include <iostream>

class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::string name;

public:
    Timer(const std::string& label);
    ~Timer();

};


#endif //LIBCUBWT_TIMER_CUH
