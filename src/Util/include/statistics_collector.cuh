#pragma once

#include <chrono>
#include <cuda_runtime.h>
#include <string>
#include <iomanip>
#include <sstream>
#include <sys/resource.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>

/**
 * Statistics collector for comprehensive benchmarking
 * Tracks: wall time, SA time, LZ77 time, transfer times, host RAM, GPU RAM
 */
class StatisticsCollector {
private:
    std::chrono::high_resolution_clock::time_point wall_start;
    std::chrono::high_resolution_clock::time_point wall_end;
    
    double sa_construction_time_ms = 0.0;
    double lz77_processing_time_ms = 0.0;
    double transfer_time_h2d_ms = 0.0;  // Host to Device
    double transfer_time_d2h_ms = 0.0;  // Device to Host
    
    size_t peak_host_ram_bytes = 0;
    size_t peak_gpu_ram_bytes = 0;
    size_t initial_gpu_free_bytes = 0;
    size_t initial_gpu_total_bytes = 0;
    
    bool sa_time_set = false;
    bool lz77_time_set = false;
    
    // Helper to get current process RSS (Resident Set Size) in bytes
    // Linux-only: reads from /proc/self/status (standard on Linux systems)
    size_t getCurrentRSS() {
        FILE* file = fopen("/proc/self/status", "r");
        if (!file) return 0;
        
        size_t rss = 0;
        char line[128];
        while (fgets(line, sizeof(line), file)) {
            if (strncmp(line, "VmRSS:", 6) == 0) {
                sscanf(line, "VmRSS: %zu kB", &rss);
                rss *= 1024;  // Convert KB to bytes
                break;
            }
        }
        fclose(file);
        return rss;
    }
    
    void updatePeakHostRAM() {
        size_t current = getCurrentRSS();
        if (current > peak_host_ram_bytes) {
            peak_host_ram_bytes = current;
        }
    }
    
    void updatePeakGPURAM() {
        size_t free, total;
        cudaMemGetInfo(&free, &total);
        size_t used = total - free;
        if (used > peak_gpu_ram_bytes) {
            peak_gpu_ram_bytes = used;
        }
    }

public:
    StatisticsCollector() {
        wall_start = std::chrono::high_resolution_clock::now();
        cudaMemGetInfo(&initial_gpu_free_bytes, &initial_gpu_total_bytes);
        peak_host_ram_bytes = getCurrentRSS();
        peak_gpu_ram_bytes = initial_gpu_total_bytes - initial_gpu_free_bytes;
    }
    
    void recordSATime(double ms) {
        sa_construction_time_ms = ms;
        sa_time_set = true;
        updatePeakHostRAM();
        updatePeakGPURAM();
    }
    
    void recordLZ77Time(double ms) {
        lz77_processing_time_ms = ms;
        lz77_time_set = true;
        updatePeakHostRAM();
        updatePeakGPURAM();
    }
    
    void recordTransferH2D(double ms) {
        transfer_time_h2d_ms += ms;
        updatePeakGPURAM();
    }
    
    void recordTransferD2H(double ms) {
        transfer_time_d2h_ms += ms;
        updatePeakHostRAM();
        updatePeakGPURAM();
    }
    
    void updateMemoryStats() {
        updatePeakHostRAM();
        updatePeakGPURAM();
    }
    
    void finalize() {
        wall_end = std::chrono::high_resolution_clock::now();
        updatePeakHostRAM();
        updatePeakGPURAM();
    }
    
    double getWallTimeMs() const {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(wall_end - wall_start);
        return duration.count();
    }
    
    double getWallTimeSec() const {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(wall_end - wall_start);
        return duration.count() / 1000.0;
    }
    
    void printSummary() const {
        double wall_time_ms = getWallTimeMs();
        double wall_time_sec = getWallTimeSec();
        
        std::cout << "\n";
        std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║              LZ77 GPU COMPRESSION STATISTICS                  ║\n";
        std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
        
        // Timing Section
        std::cout << "║ TIMING                                                         ║\n";
        std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "║  Wall Time (Total):                    " << std::setw(12) << wall_time_ms << " ms  ║\n";
        std::cout << "║                                       " << std::setw(12) << wall_time_sec << " s   ║\n";
        
        if (sa_time_set) {
            std::cout << "║  SA Construction:                     " << std::setw(12) << sa_construction_time_ms << " ms  ║\n";
        }
        
        if (lz77_time_set) {
            std::cout << "║  LZ77 Processing:                     " << std::setw(12) << lz77_processing_time_ms << " ms  ║\n";
        }
        
        double other_time = wall_time_ms - sa_construction_time_ms - lz77_processing_time_ms;
        if (other_time > 0) {
            std::cout << "║  Other Operations:                    " << std::setw(12) << other_time << " ms  ║\n";
        }
        
        std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
        
        // Transfer Section
        std::cout << "║ DATA TRANSFER                                                  ║\n";
        std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║  Host → GPU (H2D):                     " << std::setw(12) << transfer_time_h2d_ms << " ms  ║\n";
        std::cout << "║  GPU → Host (D2H):                     " << std::setw(12) << transfer_time_d2h_ms << " ms  ║\n";
        double total_transfer = transfer_time_h2d_ms + transfer_time_d2h_ms;
        std::cout << "║  Total Transfer Time:                   " << std::setw(12) << total_transfer << " ms  ║\n";
        if (wall_time_ms > 0) {
            double transfer_percent = (total_transfer / wall_time_ms) * 100.0;
            std::cout << "║  Transfer Overhead:                    " << std::setw(11) << transfer_percent << " %   ║\n";
        }
        
        std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
        
        // Memory Section
        std::cout << "║ MEMORY USAGE                                                  ║\n";
        std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
        std::cout << std::setprecision(2);
        std::cout << "║  Host RAM (Peak):                      " << std::setw(12) << (peak_host_ram_bytes / (1024.0 * 1024.0)) << " MB  ║\n";
        std::cout << "║                                       " << std::setw(12) << (peak_host_ram_bytes / (1024.0 * 1024.0 * 1024.0)) << " GB  ║\n";
        std::cout << "║  GPU RAM (Peak):                      " << std::setw(12) << (peak_gpu_ram_bytes / (1024.0 * 1024.0)) << " MB  ║\n";
        std::cout << "║                                       " << std::setw(12) << (peak_gpu_ram_bytes / (1024.0 * 1024.0 * 1024.0)) << " GB  ║\n";
        std::cout << "║  GPU RAM (Total Available):            " << std::setw(12) << (initial_gpu_total_bytes / (1024.0 * 1024.0 * 1024.0)) << " GB  ║\n";
        std::cout << "║  GPU RAM (Initial Free):               " << std::setw(12) << (initial_gpu_free_bytes / (1024.0 * 1024.0 * 1024.0)) << " GB  ║\n";
        
        std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
        std::cout << std::endl;
    }
    
    // CSV output for easy parsing
    void printCSV() const {
        double wall_time_ms = getWallTimeMs();
        double total_transfer = transfer_time_h2d_ms + transfer_time_d2h_ms;
        
        std::cout << "\n=== CSV Statistics ===" << std::endl;
        std::cout << "wall_time_ms,sa_construction_ms,lz77_processing_ms,"
                  << "transfer_h2d_ms,transfer_d2h_ms,total_transfer_ms,"
                  << "host_ram_mb,host_ram_gb,gpu_ram_mb,gpu_ram_gb" << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << wall_time_ms << ","
                  << sa_construction_time_ms << ","
                  << lz77_processing_time_ms << ","
                  << transfer_time_h2d_ms << ","
                  << transfer_time_d2h_ms << ","
                  << total_transfer << ","
                  << (peak_host_ram_bytes / (1024.0 * 1024.0)) << ","
                  << (peak_host_ram_bytes / (1024.0 * 1024.0 * 1024.0)) << ","
                  << (peak_gpu_ram_bytes / (1024.0 * 1024.0)) << ","
                  << (peak_gpu_ram_bytes / (1024.0 * 1024.0 * 1024.0)) << std::endl;
    }
};
