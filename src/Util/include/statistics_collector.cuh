#pragma once

#include <chrono>
#include <cuda_runtime.h>
#include <string>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <sys/resource.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>

/**
 * Statistics collector for comprehensive benchmarking
 * Tracks: wall time, SA time, LZ77 time (GPU compute, CPU compute, transfers), 
 *         transfer times, host RAM, GPU RAM
 */
class StatisticsCollector {
private:
    std::chrono::high_resolution_clock::time_point wall_start;
    std::chrono::high_resolution_clock::time_point wall_end;
    
    double sa_construction_time_ms = 0.0;
    
    // LZ77 timing breakdown
    double lz77_gpu_computation_time_ms = 0.0;  // GPU phases 1-3 (PSV/NSV computation)
    double lz77_cpu_computation_time_ms = 0.0;   // CPU ComputeLZ77 function
    double lz77_transfer_time_ms = 0.0;          // Data transfers during LZ77 phase (H2D + D2H)
    
    double lz77_processing_time_ms = 0.0;
    
    // Global transfer times (all phases combined)
    double transfer_time_h2d_ms = 0.0;  // Host to Device
    double transfer_time_d2h_ms = 0.0;   // Device to Host
    
    // LZ77-specific transfer times
    double lz77_transfer_h2d_ms = 0.0;
    double lz77_transfer_d2h_ms = 0.0;
    
    size_t peak_host_ram_bytes = 0;
    size_t peak_gpu_ram_bytes = 0;
    size_t initial_gpu_free_bytes = 0;
    size_t initial_gpu_total_bytes = 0;
    
    bool sa_time_set = false;
    bool lz77_time_set = false;
    bool lz77_gpu_time_set = false;
    
    // Helper to get current process RSS (Resident Set Size) in bytes
    // Linux-only: reads from /proc/self/status (standard on Linux systems)
    size_t getCurrentRSS() {
        FILE* file = fopen("/proc/self/status", "r");
        if (!file) return 0;
        
        size_t rss = 0;
        char line[128];
        while (fgets(line, sizeof(line), file)) {
            if (strncmp(line, "VmRSS:", 6) == 0) {
                if (sscanf(line, "VmRSS: %zu kB", &rss) == 1) {
                    rss *= 1024;  // Convert KB to bytes
                }
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
    
    static const int TABLE_WIDTH = 65;

    void printSectionHeader(const std::string& title) const {
        int padding = TABLE_WIDTH - 2 - static_cast<int>(title.length()) - 1;  // "| " + title + "|"
        if (padding < 0) padding = 0;
        std::cout << "| " << title << std::string(padding, ' ') << "|\n";
    }

    void printFormattedLine(const std::string& label, double value, const std::string& unit) const {
        const int LABEL_START = 2;  // "| "
        const int NUMBER_WIDTH = 12;
        const int UNIT_WIDTH = 5;   // " ms |" or " s  |" (one space before |)
        
        int label_width = label.length();
        int available_width = TABLE_WIDTH - LABEL_START - NUMBER_WIDTH - UNIT_WIDTH;
        int padding = available_width - label_width;
        
        std::cout << "| " << label;
        if (padding > 0) {
            std::cout << std::string(padding, ' ');
        }
        std::cout << std::right << std::setw(NUMBER_WIDTH) << std::fixed << std::setprecision(2) << value;
        if (unit.length() == 1) {
            std::cout << " " << unit << "  |\n";
        } else {
            std::cout << " " << unit << " |\n";
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
        lz77_processing_time_ms = ms;  // Total LZ77 processing time
        lz77_time_set = true;
        updatePeakHostRAM();
        updatePeakGPURAM();
    }
    
    void recordLZ77CPUTime(double ms) {
        lz77_cpu_computation_time_ms = ms;
        updatePeakHostRAM();
        updatePeakGPURAM();
    }
    
    void recordLZ77GPUTime(double ms) {
        lz77_gpu_computation_time_ms = ms;
        lz77_gpu_time_set = true;
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
    
    // Record LZ77-specific transfers (separate from SA phase transfers)
    void recordLZ77TransferH2D(double ms) {
        lz77_transfer_h2d_ms += ms;
        lz77_transfer_time_ms += ms;
        transfer_time_h2d_ms += ms;  // Also add to global total
        updatePeakGPURAM();
    }
    
    void recordLZ77TransferD2H(double ms) {
        lz77_transfer_d2h_ms += ms;
        lz77_transfer_time_ms += ms;
        transfer_time_d2h_ms += ms;  // Also add to global total
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
        std::cout << "+===============================================================+\n";
        std::cout << "|              LZ77 GPU COMPRESSION STATISTICS                  |\n";
        std::cout << "+===============================================================+\n";
        
        // Timing Section
        printSectionHeader("TIMING");
        std::cout << "+---------------------------------------------------------------+\n";
        printFormattedLine("Wall Time (Total):", wall_time_ms, "ms");
        printFormattedLine("", wall_time_sec, "s");
        
        if (sa_time_set) {
            printFormattedLine("SA Construction:", sa_construction_time_ms, "ms");
        }
        
        // LZ77 breakdown
        if (lz77_time_set) {
            printFormattedLine("LZ77 Processing:", lz77_processing_time_ms, "ms");
            if (lz77_gpu_time_set) {
                printFormattedLine("  - GPU Computation:", lz77_gpu_computation_time_ms, "ms");
            }
            if (lz77_cpu_computation_time_ms > 0) {
                printFormattedLine("  - CPU Computation:", lz77_cpu_computation_time_ms, "ms");
            }
            if (lz77_transfer_time_ms > 0) {
                printFormattedLine("  - Data Transfer:", lz77_transfer_time_ms, "ms");
            }
        }
        
        double total_tracked = sa_construction_time_ms + lz77_processing_time_ms;
        double remainder_time = wall_time_ms - total_tracked;
        if (remainder_time > 0) {
            printFormattedLine("Remainder Time:", remainder_time, "ms");
        }
        
        std::cout << "+---------------------------------------------------------------+\n";
        
        // Transfer Section
        printSectionHeader("DATA TRANSFER");
        std::cout << "+---------------------------------------------------------------+\n";
        printFormattedLine("Host -> GPU (H2D):", transfer_time_h2d_ms, "ms");
        printFormattedLine("GPU -> Host (D2H):", transfer_time_d2h_ms, "ms");
        double total_transfer = transfer_time_h2d_ms + transfer_time_d2h_ms;
        printFormattedLine("Total Transfer Time:", total_transfer, "ms");
        if (wall_time_ms > 0) {
            double transfer_percent = (total_transfer / wall_time_ms) * 100.0;
            printFormattedLine("Transfer Overhead:", transfer_percent, "%");
        }
        
        std::cout << "+---------------------------------------------------------------+\n";
        
        // Memory Section
        printSectionHeader("MEMORY USAGE");
        std::cout << "+---------------------------------------------------------------+\n";
        printFormattedLine("Host RAM (Peak):", peak_host_ram_bytes / (1024.0 * 1024.0), "MB");
        printFormattedLine("", peak_host_ram_bytes / (1024.0 * 1024.0 * 1024.0), "GB");
        printFormattedLine("GPU RAM (Peak):", peak_gpu_ram_bytes / (1024.0 * 1024.0), "MB");
        printFormattedLine("", peak_gpu_ram_bytes / (1024.0 * 1024.0 * 1024.0), "GB");
        printFormattedLine("GPU RAM (Total Available):", initial_gpu_total_bytes / (1024.0 * 1024.0 * 1024.0), "GB");
        printFormattedLine("GPU RAM (Initial Free):", initial_gpu_free_bytes / (1024.0 * 1024.0 * 1024.0), "GB");
        
        std::cout << "+===============================================================+\n";
        std::cout << std::endl;
    }
    
    // CSV output for easy parsing
    void printCSV() const {
        double wall_time_ms = getWallTimeMs();
        double total_transfer = transfer_time_h2d_ms + transfer_time_d2h_ms;
        
        std::cout << "\n=== CSV Statistics ===" << std::endl;
        std::cout << "wall_time_ms,sa_construction_ms,"
                  << "lz77_processing_ms,lz77_gpu_computation_ms,lz77_cpu_computation_ms,lz77_transfer_ms,"
                  << "transfer_h2d_ms,transfer_d2h_ms,total_transfer_ms,"
                  << "host_ram_mb,host_ram_gb,gpu_ram_mb,gpu_ram_gb" << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << wall_time_ms << ","
                  << sa_construction_time_ms << ","
                  << lz77_processing_time_ms << ","
                  << lz77_gpu_computation_time_ms << ","
                  << lz77_cpu_computation_time_ms << ","
                  << lz77_transfer_time_ms << ","
                  << transfer_time_h2d_ms << ","
                  << transfer_time_d2h_ms << ","
                  << total_transfer << ","
                  << (peak_host_ram_bytes / (1024.0 * 1024.0)) << ","
                  << (peak_host_ram_bytes / (1024.0 * 1024.0 * 1024.0)) << ","
                  << (peak_gpu_ram_bytes / (1024.0 * 1024.0)) << ","
                  << (peak_gpu_ram_bytes / (1024.0 * 1024.0 * 1024.0)) << std::endl;
    }
};
