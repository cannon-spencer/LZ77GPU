#pragma once

#include <chrono>

inline uint64_t g_alloc_time_ns = 0;
inline uint64_t g_sort_time_ns = 0;
inline uint64_t g_kernel_diff_time_ns = 0;
inline uint64_t g_scan_time_ns = 0;
inline uint64_t g_kernel_assign_time_ns = 0;
inline uint64_t g_copy_time_ns = 0;
inline uint64_t g_cleanup_time_ns = 0;

inline auto now() {
    return std::chrono::high_resolution_clock::now();
}

inline void record_time(uint64_t& accumulator,
                        const std::chrono::high_resolution_clock::time_point& start) {
    auto end = std::chrono::high_resolution_clock::now();
    accumulator += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}
