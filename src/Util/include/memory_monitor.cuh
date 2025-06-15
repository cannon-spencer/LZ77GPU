#pragma once
#include "cuda_utils.cuh"
#include <atomic>
#include <chrono>
#include <thread>


/**
 * MEMORY MONITOR FOR TRACKING PEAK GPU MEMORY USAGE
 */
class MemoryMonitor {
private:
    std::thread monitoring_thread;
    std::atomic<bool> stop_monitoring;
    size_t baselineFreeMem = 0;
    size_t peakUsage = 0;
    constexpr static double BYTES_PER_MB = 1024.0 * 1024.0;

    void init() {
        size_t totalMem;
        CHECK_CUDA_ERROR(cudaMemGetInfo(&baselineFreeMem, &totalMem));
        peakUsage = 0;
    }

    void update_peak_usage() {
        size_t currentFree, totalMem;
        CHECK_CUDA_ERROR(cudaMemGetInfo(&currentFree, &totalMem));
        size_t usedNow = (baselineFreeMem > currentFree) ? (baselineFreeMem - currentFree) : 0;
        if (usedNow > peakUsage) {
            peakUsage = usedNow;
        }
    }

    void monitor() {
        while (!stop_monitoring.load()) {
            update_peak_usage();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

public:
    MemoryMonitor() : stop_monitoring(false) {}

    void start() {
        init();
        stop_monitoring.store(false);
        monitoring_thread = std::thread(&MemoryMonitor::monitor, this);
    }

    void stop() {
        stop_monitoring.store(true);
        if (monitoring_thread.joinable()) {
            monitoring_thread.join();
        }
    }

    double get_peak_usage_mb() const {
        return static_cast<double>(peakUsage) / BYTES_PER_MB;
    }

    ~MemoryMonitor() {
        stop();
    }
};
