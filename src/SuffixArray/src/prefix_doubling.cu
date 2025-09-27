#include <vector>
#include <cstdint>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include "cuda_utils.cuh"
#include "profiler.cuh"

template<typename T>
__global__
void init_index_rank_kernel_template(const uint8_t* d_s, T* d_index, T* d_rank, size_t n){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        d_index[i] = static_cast<T>(i);
        d_rank[i]  = static_cast<T>(d_s[i]);
    }
}

template<typename T>
__global__
void pack_keys_kernel_template(const T* d_rank, size_t n, size_t k, T* key_hi, T* key_lo){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        key_hi[i] = d_rank[i];
        key_lo[i] = (i + k < n) ? d_rank[i + k] : static_cast<T>(0);
    }
}

/**
 * Kernel to compute the "diff" array by comparing consecutive sorted suffix indices:
 * If sorted suffix i differs from suffix i-1, set diff[i] = 1 else 0.
 * The first suffix i=0 is always a new group => diff[0] = 1.
 */
template<typename T>
__global__
void compute_diff_kernel_template(const T* d_index, const T* d_rank, size_t n, size_t k, T* d_diff){
    T i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (i == 0) {
            d_diff[0] = 1; // first always starts a new group
        } else {
            // previous suffix index
            T prev = d_index[i - 1];
            // current suffix index
            T curr = d_index[i];

            // Compare (rank[prev], rank[prev+k]) vs (rank[curr], rank[curr+k])
            T r1_prev = d_rank[prev];
            T r2_prev = ((prev + k) < n) ? d_rank[prev + k] : 0;
            T r1_curr = d_rank[curr];
            T r2_curr = ((curr + k) < n) ? d_rank[curr + k] : 0;

            bool diff = (r1_curr != r1_prev) || (r2_curr != r2_prev);
            d_diff[i] = diff ? 1 : 0;
        }
    }
}

/**
 * Kernel to assign new ranks from the inclusive-scan "group ID" (d_diff).
 * The sorted order is in d_index, so suffix i in sorted order => d_index[i].
 * We'll do: rank[d_index[i]] = d_diff[i].
 */
template<typename T>
__global__
void assign_ranks_kernel_template(const T* d_index, const T* d_diff, T* d_rank, size_t n){
    T i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        T suffix = d_index[i];
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

template<typename T>
T* build_suffix_array_prefix_doubling_template(const std::vector<uint8_t>& s, size_t& out_length){
    size_t n = s.size();
    if (n == 0) {
        out_length = 0;
        return nullptr;
    }

    out_length = n;

    // Kernel config
    int blockSize = 1024;
    int gridSize = static_cast<int>((n + blockSize - 1) / blockSize);

    // Allocate device arrays
    T* d_rank = nullptr;
    T* d_index = nullptr;
    T* d_diff = nullptr;

    auto t0 = now();
    myCudaMalloc(&d_rank, n * sizeof(T), "d_rank");
    myCudaMalloc(&d_index, n * sizeof(T), "d_index");

    // Upload input s[] to GPU
    uint8_t* d_s = nullptr;
    cudaMalloc(&d_s, n * sizeof(uint8_t));
    CHECK_CUDA_ERROR(cudaMemcpy(d_s, s.data(), n * sizeof(uint8_t), cudaMemcpyHostToDevice));

    // // Initialize the index and rank array using thrust
    // cudaMalloc(&d_s, n * sizeof(uint8_t));
    // CHECK_CUDA_ERROR(cudaMemcpy(d_s, s.data(), n * sizeof(uint8_t), cudaMemcpyHostToDevice));

    init_index_rank_kernel_template<<<gridSize, blockSize>>>(d_s, d_index, d_rank, n);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    cudaFree(d_s);

    // Allocate diff after freeing input array
    myCudaMalloc(&d_diff, n * sizeof(T), "d_diff");
    thrust::device_vector<T> d_key_lo(n);

    std::cout << "Total GPU Memory Allocated for " << typeid(T).name() << ": " 
              << g_allocated / (1024.0 * 1024.0) << " MB\n";
    record_time(g_alloc_time_ns, t0);

    // Prefix doubling
    for (size_t k = 1; k < n; k <<= 1) {
        // 1) Sort d_index by (rank[i], rank[i+k])
        {
            auto t2 = now();
            pack_keys_kernel_template<<<gridSize, blockSize>>>(
            d_rank, n, k,
            d_diff,
            thrust::raw_pointer_cast(d_key_lo.data()));
            CHECK_CUDA_ERROR(cudaGetLastError());
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());

            auto key_begin = thrust::make_zip_iterator(
            thrust::make_tuple(thrust::device_pointer_cast(d_diff), d_key_lo.begin()));
            auto key_end = key_begin + n;

            thrust::sort_by_key(thrust::device,
                            key_begin, key_end,
                            thrust::device_pointer_cast(d_index));


            // SuffixComparatorTemplate<T> cmp(d_rank, k, n);
            // thrust::device_ptr<T> d_index_ptr = thrust::device_pointer_cast(d_index);
            // thrust::sort(thrust::device, d_index_ptr, d_index_ptr + n, cmp);
            record_time(g_sort_time_ns, t2);
        }

        // 2) compute diff array
        auto t3 = now();
        compute_diff_kernel_template<<<gridSize, blockSize>>>(d_index, d_rank, n, k, d_diff);

        // 3) inclusive scan => group IDs
        {
            thrust::device_ptr<T> d_diff_ptr = thrust::device_pointer_cast(d_diff);
            thrust::inclusive_scan(d_diff_ptr, d_diff_ptr + n, d_diff_ptr);
        }

        // 4) assign new ranks => rank[index[i]] = d_diff[i]
        assign_ranks_kernel_template<<<gridSize, blockSize>>>(d_index, d_diff, d_rank, n);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        record_time(g_kernel_assign_time_ns, t3);

        // 5) check if all ranks are distinct => if d_diff[n-1] == n
        auto t6 = now();
        T max_rank;
        CHECK_CUDA_ERROR(cudaMemcpy(&max_rank, d_diff + (n - 1), sizeof(T), cudaMemcpyDeviceToHost));
        record_time(g_copy_time_ns, t6);

        // all ranks are distinct -> done
        if (max_rank == static_cast<T>(n)) {
            break;
        }
    }

    // Clean up
    auto t8 = now();
    cudaFree(d_rank);
    cudaFree(d_diff);
    record_time(g_cleanup_time_ns, t8);

    return d_index;
}
uint32_t* build_suffix_array_prefix_doubling(const std::vector<uint8_t>& s, size_t& out_length) {
    return build_suffix_array_prefix_doubling_template<uint32_t>(s, out_length);
}

size_t* build_suffix_array_prefix_doubling_64(const std::vector<uint8_t>& s, size_t& out_length) {
    return build_suffix_array_prefix_doubling_template<size_t>(s, out_length);
}