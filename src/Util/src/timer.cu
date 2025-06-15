#include "timer.cuh"

Timer::Timer(const std::string& label) : name(label) {
    start_time = std::chrono::high_resolution_clock::now();
}

Timer::~Timer() {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count();
    std::cout << name << " took " << duration << " ms" << std::endl;
}
