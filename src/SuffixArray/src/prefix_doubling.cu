#include <vector>
#include <cstdint>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/tuple.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>

#include "cuda_utils.cuh"
#include "profiler.cuh"

/**
 * Build suffix array with prefix doubling, no SuffixKey array.
 * We'll store:
 *   - d_rank[n]  - the rank array
 *   - d_index[n] - the suffix ordering
 *
 * Steps per iteration:
 *   1) Build packed keys: key[i] = pack(R[i], R[i+k]). Values = d_index (suffix i).
 *   2) Run thrust::sort_by_key(keys, d_index).
 *   3) inclusive_scan of on-the-fly head flags, writing directly to d_rank[d_index[i]].
 *   4) if max rank == n, break early
 */
uint32_t* build_suffix_array_prefix_doubling_device(const std::vector<uint8_t>& s){
    size_t n = s.size();
    if (n == 0) return nullptr;

    // Allocate device arrays
    uint64_t* d_keys  = nullptr;
    uint32_t* d_rank  = nullptr;
    uint32_t* d_index = nullptr;

    SA_DEBUG_START(t0);
    myCudaMalloc(&d_rank,  n * sizeof(uint32_t), "d_rank");
    myCudaMalloc(&d_index, n * sizeof(uint32_t), "d_index");
    myCudaMalloc(&d_keys,  n * sizeof(uint64_t), "d_keys");

    // Upload input s[] to GPU
    uint8_t* d_s = nullptr;
    cudaMalloc(&d_s, n * sizeof(uint8_t));
    CHECK_CUDA_ERROR(cudaMemcpy(d_s, s.data(), n * sizeof(uint8_t), cudaMemcpyHostToDevice));

    // Initialize d_index = [0..n-1]
    {
        auto d_index_ptr = thrust::device_pointer_cast(d_index);
        thrust::sequence(thrust::device, d_index_ptr, d_index_ptr + n, 0);
    }

    // Initialize d_rank from input string (promote to uint32 + 1)
    {
        auto s_ptr = thrust::device_pointer_cast(d_s);
        auto d_rank_ptr = thrust::device_pointer_cast(d_rank);
        thrust::transform(thrust::device,
                          s_ptr, s_ptr + n,
                          d_rank_ptr,
                          [] __device__ (uint8_t c) {
            return static_cast<uint32_t>(c) + 1;
        });
    }
    cudaFreeAsync(d_s, 0);

    SA_DEBUG_LOG("Total GPU Memory Allocated: " << (g_allocated / (1024.0 * 1024.0)) << " MB");
    SA_DEBUG_END(g_init_time_ns, t0);

    // Prefix doubling
    for (size_t k = 1; k < n; k <<= 1) {
        // Values must be "suffix i" for the keys built for i this round.
        {
            SA_DEBUG_START(t1);
            auto d_index_ptr = thrust::device_pointer_cast(d_index);
            thrust::sequence(thrust::device, d_index_ptr, d_index_ptr + n, 0);
            SA_DEBUG_END(g_index_seed_time_ns, t1);
        }

        // 1) Build packed keys = (R[i], R[i+k]) and values = index, then radix sort_by_key
        {
            // define a range of sequential values
            thrust::counting_iterator<uint32_t> I(0);

            // pointers to key/value pairs
            auto d_keys_ptr  = thrust::device_pointer_cast(d_keys);
            auto d_index_ptr = thrust::device_pointer_cast(d_index);

            // Build keys with a transform – recompute every round from current ranks and k
            SA_DEBUG_START(t2);
            thrust::transform(
                    I, I + n, d_keys_ptr,
                    [R = d_rank, n, k] __device__ (uint32_t i) {
                        uint32_t a = R[i];
                        uint32_t b = (i + k < n) ? R[i + k] : 0u;
                        return (uint64_t(a) << 32) | uint64_t(b);
                    });
            SA_DEBUG_END(g_build_keys_time_ns, t2);

            // 2) Radix sort_by_key: keys determine order, d_index rides along
            SA_DEBUG_START(t3);
            thrust::sort_by_key(d_keys_ptr, d_keys_ptr + n, d_index_ptr);
            SA_DEBUG_END(g_sort_time_ns, t3);
        }

        // 3) inclusive scan of head flags => new ranks directly in-place
        auto counting_begin = thrust::make_counting_iterator<uint32_t>(0);
        SA_DEBUG_START(t4);
        auto head_flags_begin = thrust::make_transform_iterator(
                counting_begin,
                [keys = d_keys] __device__ (uint32_t i) {
                    if (i == 0) return 1u;
                    return (keys[i] != keys[i - 1]) ? 1u : 0u;
                });
        auto head_flags_end = head_flags_begin + n;
        auto d_rank_ptr = thrust::device_pointer_cast(d_rank);
        auto d_index_ptr = thrust::device_pointer_cast(d_index);
        auto rank_scatter_begin = thrust::make_permutation_iterator(d_rank_ptr, d_index_ptr);
        SA_DEBUG_END(g_diff_time_ns, t4);

        SA_DEBUG_START(t5);
        thrust::inclusive_scan(thrust::device, head_flags_begin, head_flags_end, rank_scatter_begin);
        SA_DEBUG_END(g_scan_time_ns, t5);

        // 5) check if all ranks are distinct => if d_diff[n-1] == n
        SA_DEBUG_START(t7);
        uint32_t max_rank = 0;
        thrust::copy_n(rank_scatter_begin + (n - 1), 1, &max_rank);
        SA_DEBUG_END(g_copy_chk_time_ns, t7);

        // all ranks are distinct -> done
        if (max_rank == static_cast<uint32_t>(n)) {
            break;
        }
    }

    // Clean up
    cudaFreeAsync(d_rank, 0);
    cudaFreeAsync(d_keys, 0);

    return d_index; // caller owns d_index and must free
}



// Public wrapper: Returns the host Suffix Array
std::vector<uint32_t> build_suffix_array_prefix_doubling(const std::vector<uint8_t>& s)
{
    std::vector<uint32_t> host_sa;
    if (s.empty()) return host_sa;

    // Compute SA on device
    SA_DEBUG_START(t0);
    uint32_t* d_sa = build_suffix_array_prefix_doubling_device(s);
    SA_DEBUG_END(g_compute_sa_time_ns, t0);

    // Allocate host memory & transfer SA back to host (this is kind of slow)
    SA_DEBUG_START(t1);
    host_sa.resize(s.size());
    CHECK_CUDA_ERROR(cudaMemcpy(host_sa.data(), d_sa, s.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    SA_DEBUG_END(g_copy_back_time_ns, t1);

    cudaFreeAsync(d_sa, 0);

    return host_sa;
}
