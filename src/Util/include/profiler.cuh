#pragma once

#include <chrono>

inline uint64_t g_init_time_ns;          // H2D + initial sequence/transform
inline uint64_t g_build_keys_time_ns;    // thrust::transform building keys
inline uint64_t g_sort_time_ns;          // sort_by_key only
inline uint64_t g_diff_time_ns;          // compute_diff_from_keys kernel
inline uint64_t g_scan_time_ns;          // inclusive_scan
inline uint64_t g_assign_time_ns;        // assign_ranks kernel
inline uint64_t g_copy_chk_time_ns;      // copy back max rank (termination check)
inline uint64_t g_cleanup_time_ns;       // frees
inline uint64_t g_copy_back_time_ns;     // final SA copy D2H


inline auto now() {
    return std::chrono::high_resolution_clock::now();
}

inline void record_time(uint64_t& accumulator,
                        const std::chrono::high_resolution_clock::time_point& start) {
    auto end = std::chrono::high_resolution_clock::now();
    accumulator += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}
