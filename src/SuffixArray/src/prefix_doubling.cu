#include <vector>
#include <cstdint>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/iterator/counting_iterator.h>

#include "cuda_utils.cuh"
#include "profiler.cuh"



/**
 * CUDA kernel to initialize the rank and index arrays for suffix array construction.
 *
 * - Converts the input string `s` (8-bit characters) into 32-bit initial ranks.
 * - Initializes the index array with values [0, 1, 2, ..., n-1].
 * */

__global__
void initialize_rank_and_index(const uint8_t* s, uint32_t* d_rank, uint32_t* d_index, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        d_rank[i] = static_cast<uint32_t>(s[i]) + 1;
        d_index[i] = i;
    }
}


/**
 * Kernel to compute the "diff" array by comparing consecutive sorted suffix indices:
 * If sorted suffix i differs from suffix i-1, set diff[i] = 1 else 0.
 * The first suffix i=0 is always a new group => diff[0] = 1.
 */
__global__
void compute_diff_from_keys(const uint64_t* keys, uint32_t* diff, size_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        diff[i] = (i == 0) ? 1u : (keys[i] != keys[i - 1]);
    }
}


/**
 * Kernel to assign new ranks from the inclusive-scan "group ID" (d_diff).
 * The sorted order is in d_index, so suffix i in sorted order => d_index[i].
 * We'll do: rank[d_index[i]] = d_diff[i].
 */
__global__
void assign_ranks_kernel(const uint32_t* d_index, const uint32_t* d_diff, uint32_t* d_rank, size_t n){
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        uint32_t suffix = d_index[i];
        d_rank[suffix] = d_diff[i];
    }
}


/**
 * Build suffix array with prefix doubling, no SuffixKey array.
 * We'll store:
 *   - d_rank[n]  - the rank array
 *   - d_index[n] - the suffix ordering
 *   - d_diff[n]  - difference array, scanned to get group IDs
 *
 * Steps per iteration:
 *   1) Sort d_index by comparing (rank[i], rank[i+k]).
 *   2) compute_diff_kernel => d_diff[i] in {0,1}.
 *   3) inclusive_scan(d_diff).
 *   4) assign_ranks_kernel => rank[suffixIndex] = groupID
 *   5) if d_diff[n-1] == n, break early
 */

// todo: make this return a device pointer so we can utilize it without pulling it off of the GPU
std::vector<uint32_t> build_suffix_array_prefix_doubling(const std::vector<uint8_t>& s){
    size_t n = s.size();
    if (n == 0) return {};

    // Kernel config
    int blockSize = 1024;
    int gridSize  = static_cast<int>((n + blockSize - 1) / blockSize);

    // Allocate device arrays
    uint64_t* d_keys  = nullptr;
    uint32_t* d_rank  = nullptr;
    uint32_t* d_index = nullptr;
    uint32_t* d_diff  = nullptr;

    auto t0 = now();
    myCudaMalloc(&d_rank,  n * sizeof(uint32_t), "d_rank");
    myCudaMalloc(&d_index, n * sizeof(uint32_t), "d_index");
    myCudaMalloc(&d_diff,  n * sizeof(uint32_t), "d_diff");
    myCudaMalloc(&d_keys,  n * sizeof(uint64_t), "d_keys");

    // Upload input s[] to GPU
    uint8_t* d_s = nullptr;
    cudaMalloc(&d_s, n * sizeof(uint8_t));
    CHECK_CUDA_ERROR(cudaMemcpy(d_s, s.data(), n * sizeof(uint8_t), cudaMemcpyHostToDevice));

    // Do initializations to correct structure in parallel
    initialize_rank_and_index<<<gridSize, blockSize>>>(d_s, d_rank, d_index, n);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    cudaFree(d_s);

    std::cout << "Total GPU Memory Allocated: " << g_allocated / (1024.0 * 1024.0) << " MB\n";
    record_time(g_alloc_time_ns, t0);


    // Prefix doubling
    for (size_t k = 1; k < n; k <<= 1) {
        // 1) Build packed keys = (R[i], R[i+k]) and values = index, then radix sort_by_key
        {
            auto t1 = now();

            // define a range of sequential values
            thrust::counting_iterator<uint32_t> I(0);

            // pointers to key/value pairs
            auto d_keys_ptr  = thrust::device_pointer_cast(d_keys);
            auto d_index_ptr = thrust::device_pointer_cast(d_index);

            // Build keys with a transform – recompute every round from current ranks and k
            thrust::transform(
                I, I + n, d_keys_ptr,
                [R = d_rank, n, k] __device__ (uint32_t i) {
                        uint32_t a = R[i];
                        uint32_t b = (i + k < n) ? R[i + k] : 0u;
                        return (uint64_t(a) << 32) | uint64_t(b);
                });

            // sequence index values from 0...n on init run
            if (k == 1) thrust::sequence(d_index_ptr, d_index_ptr + n, 0);

            // Radix sort keys - d_index follows sort operations
            thrust::sort_by_key(d_keys_ptr, d_keys_ptr + n, d_index_ptr);
            record_time(g_sort_time_ns, t1);
        }

        // 2) compute head-flags (diff) directly from sorted keys
        auto t3 = now();
        compute_diff_from_keys<<<gridSize, blockSize>>>(d_keys, d_diff, n);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        record_time(g_kernel_diff_time_ns, t3);

        // 3) inclusive scan => group IDs
        {
            auto t4 = now();
            thrust::device_ptr<uint32_t> d_diff_ptr = thrust::device_pointer_cast(d_diff);
            thrust::inclusive_scan(d_diff_ptr, d_diff_ptr + n, d_diff_ptr);
            record_time(g_scan_time_ns, t4);
        }

        // 4) assign new ranks => rank[index[i]] = d_diff[i]
        auto t5 = now();
        assign_ranks_kernel<<<gridSize, blockSize>>>(d_index, d_diff, d_rank, n);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        record_time(g_kernel_assign_time_ns, t5);


        // 5) check if all ranks are distinct => if d_diff[n-1] == n
        auto t6 = now();
        uint32_t max_rank;
        CHECK_CUDA_ERROR(cudaMemcpy(&max_rank, d_diff + (n - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost));
        record_time(g_copy_time_ns, t6);

        // all ranks are distinct -> done
        if (max_rank == static_cast<uint32_t>(n)) {
            break;
        }
    }

    // At this point, d_index is sorted suffix array
    // Copy it back to host
    auto t7 = now();
    std::vector<uint32_t> hostIndex(n);
    CHECK_CUDA_ERROR(cudaMemcpy(hostIndex.data(), d_index,n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    record_time(g_copy_time_ns, t7);

    // Clean up
    auto t8 = now();
    cudaFree(d_rank);
    cudaFree(d_index);
    cudaFree(d_diff);
    cudaFree(d_keys);
    record_time(g_cleanup_time_ns, t8);

    return hostIndex;
}