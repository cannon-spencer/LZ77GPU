#pragma once

#include <chrono>

#ifndef SA_DEBUG
    #define SA_DEBUG 1
#endif

#ifdef SA_DEBUG
    #define SA_DEBUG_START(var)                      auto var = now()
    #define SA_DEBUG_END(counter_ns, start_var)      record_time((counter_ns), (start_var))
    #define SA_DEBUG_LOG(expr)                       do { std::cout << expr << std::endl; } while (0)
#else
    // no-ops; safe to use with trailing semicolons
    #define SA_DEBUG_START(var)                      do {} while (0)
    #define SA_DEBUG_END(counter_ns, start_var)      do {} while (0)
    #define SA_DEBUG_LOG(expr)                       do {} while (0)
#endif

#ifdef SA_DEBUG
inline uint64_t g_init_time_ns;          // H2D + initial sequence/transform
inline uint64_t g_build_keys_time_ns;    // thrust::transform building keys
inline uint64_t g_sort_time_ns;          // sort_by_key only
inline uint64_t g_diff_time_ns;          // compute_diff_from_keys kernel
inline uint64_t g_scan_time_ns;          // inclusive_scan
inline uint64_t g_assign_time_ns;        // assign_ranks kernel
inline uint64_t g_copy_chk_time_ns;      // copy back max rank (termination check)
inline uint64_t g_cleanup_time_ns;       // frees
inline uint64_t g_copy_back_time_ns;     // final SA copy D2H
inline uint64_t g_index_seed_time_ns;    // per-iteration thrust::sequence
inline uint64_t g_d_sa_free_time_ns;     // cudaFree(d_sa)s
inline uint64_t g_pinned_alloc_time_ns;
inline uint64_t g_pinned_d2h_time_ns;
inline uint64_t g_pinned_to_vec_time_ns;
inline uint64_t g_compute_sa_time_ns;


inline auto now() {
    return std::chrono::high_resolution_clock::now();
}

inline void record_time(uint64_t& accumulator, const std::chrono::high_resolution_clock::time_point& start) {
    auto end = std::chrono::high_resolution_clock::now();
    accumulator += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}
#endif