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
        d_rank[i] = static_cast<uint32_t>(s[i]);
        d_index[i] = i;
    }
}


/**
 * Kernel to compute the "diff" array by comparing consecutive sorted suffix indices:
 * If sorted suffix i differs from suffix i-1, set diff[i] = 1 else 0.
 * The first suffix i=0 is always a new group => diff[0] = 1.
 */
__global__
void compute_diff_kernel(const uint32_t* d_index, const uint32_t* d_rank, size_t n, size_t k, uint32_t* d_diff){
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (i == 0) {
            d_diff[0] = 1; // first always starts a new group
        } else {
            // previous suffix index
            uint32_t prev = d_index[i - 1];
            // current suffix index
            uint32_t curr = d_index[i];

            // Compare (rank[prev], rank[prev+k]) vs (rank[curr], rank[curr+k])
            uint32_t r1_prev = d_rank[prev];
            uint32_t r2_prev = ((prev + k) < n) ? d_rank[prev + k] : 0;
            uint32_t r1_curr = d_rank[curr];
            uint32_t r2_curr = ((curr + k) < n) ? d_rank[curr + k] : 0;

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
__global__
void assign_ranks_kernel(const uint32_t* d_index, const uint32_t* d_diff, uint32_t* d_rank, size_t n){
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        uint32_t suffix = d_index[i];
        d_rank[suffix] = d_diff[i];
    }
}


__global__
void fill_shifted_ranks(const uint32_t* d_rank, uint32_t* d_rank_k, size_t k, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        d_rank_k[i] = (i + k < n) ? d_rank[i + k] : 0;
    }
}


/**
 * Custom comparator functor for suffix indices. Replaces "SuffixKey" usage.
 * Compare (i, j) by (rank[i], rank[i+k]) vs. (rank[j], rank[j+k]).
 */
struct SuffixComparator {
    const uint32_t* d_rank; // Device pointer to current rank array
    size_t k;               // offset
    size_t n;               // total length

    __host__ __device__
    SuffixComparator(const uint32_t* rank_, size_t k_, size_t n_)
            : d_rank(rank_), k(k_), n(n_) {}

    __device__
    bool operator()(uint32_t i, uint32_t j) const {
        uint32_t r1i = d_rank[i];
        uint32_t r1j = d_rank[j];
        if (r1i != r1j) return r1i < r1j;

        uint32_t r2i = (i + k < n) ? d_rank[i + k] : 0;
        uint32_t r2j = (j + k < n) ? d_rank[j + k] : 0;
        return r2i < r2j;
    }
};


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
std::vector<uint32_t> build_suffix_array_prefix_doubling(const std::vector<uint8_t>& s){
    size_t n = s.size();
    if (n == 0) return {};

    // Kernel config
    int blockSize = 1024; // 256
    int gridSize  = static_cast<int>((n + blockSize - 1) / blockSize);

    // Allocate device arrays
    uint32_t* d_rank = nullptr;
    uint32_t* d_index = nullptr;
    uint32_t* d_diff = nullptr;
    uint32_t* d_rank_k = nullptr;

    auto t0 = now();
    myCudaMalloc(&d_rank,  n * sizeof(uint32_t), "d_rank");
    myCudaMalloc(&d_index, n * sizeof(uint32_t), "d_index");

    // Upload input s[] to GPU
    uint8_t* d_s = nullptr;
    cudaMalloc(&d_s, n * sizeof(uint8_t));
    CHECK_CUDA_ERROR(cudaMemcpy(d_s, s.data(), n * sizeof(uint8_t), cudaMemcpyHostToDevice));

    // Do initializations to correct structure in parallel
    initialize_rank_and_index<<<gridSize, blockSize>>>(d_s, d_rank, d_index, n);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    cudaFree(d_s);

    // Allocate diff after freeing input array
    myCudaMalloc(&d_diff,  n * sizeof(uint32_t), "d_diff");
    myCudaMalloc(&d_rank_k,  n * sizeof(uint32_t), "d_rank_k");


    std::cout << "Total GPU Memory Allocated: " << g_allocated / (1024.0 * 1024.0) << " MB\n";
    record_time(g_alloc_time_ns, t0);


    // Prefix doubling
    for (size_t k = 1; k < n; k <<= 1) {
        // 1) Sort d_index by (rank[i], rank[i+k])
        /*{
            // Create comparator on the fly
            auto t2 = now();
            SuffixComparator cmp(d_rank, k, n);
            thrust::device_ptr <uint32_t> d_index_ptr = thrust::device_pointer_cast(d_index);
            thrust::sort(thrust::device, d_index_ptr, d_index_ptr + n, cmp);
            record_time(g_sort_time_ns, t2);
        }*/

        auto t2 = now();
        fill_shifted_ranks<<<gridSize, blockSize>>>(d_rank, d_rank_k, k, n);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        thrust::device_ptr<uint32_t> r1_ptr = thrust::device_pointer_cast(d_rank);
        thrust::device_ptr<uint32_t> r2_ptr = thrust::device_pointer_cast(d_rank_k);
        thrust::device_ptr<uint32_t> idx_ptr = thrust::device_pointer_cast(d_index);

        thrust::sort_by_key(
                thrust::make_zip_iterator(thrust::make_tuple(r1_ptr, r2_ptr)),
                thrust::make_zip_iterator(thrust::make_tuple(r1_ptr + n, r2_ptr + n)),
                idx_ptr
        );
        record_time(g_sort_time_ns, t2);

        // 2) compute diff array
        auto t3 = now();
        compute_diff_kernel<<<gridSize, blockSize>>>(d_index, d_rank, n, k, d_diff);
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
    cudaFree(d_rank_k);
    record_time(g_cleanup_time_ns, t8);

    return hostIndex;
}