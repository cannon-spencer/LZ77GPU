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
#include <thrust/host_vector.h>

#include "cuda_utils.cuh"
#include "profiler.cuh"

//static inline void print_u32_dev(const char* tag, const uint32_t* d, size_t n){
//    if (n > 64) return;
//    thrust::host_vector<uint32_t> h(n);
//    cudaMemcpy(h.data(), d, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
//    std::cout << tag << " = [";
//    for (size_t i=0;i<n;i++){ std::cout << h[i] << (i+1<n?", ":""); }
//    std::cout << "]\n";
//}
//
//static inline void print_keys_dev(const char* tag, const uint64_t* d, size_t n){
//    if (n > 64) return;
//    thrust::host_vector<uint64_t> h(n);
//    cudaMemcpy(h.data(), d, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
//    std::cout << tag << " (hi,lo) = [";
//    for (size_t i=0;i<n;i++){
//        uint32_t hi = uint32_t(h[i] >> 32);
//        uint32_t lo = uint32_t(h[i] & 0xffffffffu);
//        std::cout << "(" << hi << "," << lo << ")" << (i+1<n?", ":"");
//    }
//    std::cout << "]\n";
//}


/**
 * Kernel to compute the "diff" (head-flags) by comparing consecutive sorted KEYS:
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
 *   1) Build packed keys: key[i] = pack(R[i], R[i+k]). Values = d_index (suffix i).
 *   2) Run thrust::sort_by_key(keys, d_index).
 *   3) compute_diff_from_keys => d_diff[i] in {0,1} (head-flags where key changes).
 *   4) inclusive_scan(d_diff).
 *   5) assign_ranks_kernel => rank[suffixIndex] = groupID
 *   6) if d_diff[n-1] == n, break early
 */
uint32_t* build_suffix_array_prefix_doubling_device(const std::vector<uint8_t>& s){
    size_t n = s.size();
    if (n == 0) return nullptr;

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

    // Initialize d_index = [0..n-1]
    {
        auto d_index_ptr = thrust::device_pointer_cast(d_index);
        thrust::sequence(thrust::device, d_index_ptr, d_index_ptr + n, 0);
    }

    // Initialize d_rank from input string (promote to uint32 + 1)
    {
        auto s_ptr   = thrust::device_pointer_cast(d_s);
        auto d_rank_ptr = thrust::device_pointer_cast(d_rank);
        thrust::transform(thrust::device,
                          s_ptr, s_ptr + n,
                          d_rank_ptr,
                          [] __device__ (uint8_t c) {
            return static_cast<uint32_t>(c) + 1;
        });
    }
    cudaFree(d_s);

//    if (n <= 64) {
//        // Optional: show input as text for sanity
//        std::string s_preview(s.begin(), s.end());
//        std::cout << "DEBUG s=\"" << s_preview << "\" (n=" << n << ")\n";
//        print_u32_dev("DEBUG rank[0..n-1] (init)", d_rank, n);
//        print_u32_dev("DEBUG index[0..n-1] (init)", d_index, n);
//    }


    std::cout << "Total GPU Memory Allocated: " << g_allocated / (1024.0 * 1024.0) << " MB\n";
    record_time(g_init_time_ns, t0);


    // Prefix doubling
    for (size_t k = 1; k < n; k <<= 1) {
        // Values must be "suffix i" for the keys built for i this round.
        {
            auto d_index_ptr = thrust::device_pointer_cast(d_index);
            thrust::sequence(thrust::device, d_index_ptr, d_index_ptr + n, 0);
        }

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
            record_time(g_build_keys_time_ns, t1);

//            if (n <= 64) {
//                print_keys_dev(("DEBUG keys before sort (k=" + std::to_string(k) + ")").c_str(), d_keys, n);
//            }


            // 2) Radix sort_by_key: keys determine order, d_index rides along
            auto t2 = now();
            thrust::sort_by_key(d_keys_ptr, d_keys_ptr + n, d_index_ptr);
            record_time(g_sort_time_ns, t2);

//            if (n <= 64) {
//                print_keys_dev(("DEBUG keys after  sort (k=" + std::to_string(k) + ")").c_str(), d_keys, n);
//                print_u32_dev(("DEBUG index after sort (k=" + std::to_string(k) + ")").c_str(), d_index, n);
//            }

        }

        // 2) compute head-flags (diff) directly from sorted keys
        auto t3 = now();
        compute_diff_from_keys<<<gridSize, blockSize>>>(d_keys, d_diff, n);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        record_time(g_diff_time_ns, t3);

        // 3) inclusive scan => group IDs
        {
            auto t4 = now();
            thrust::device_ptr<uint32_t> d_diff_ptr = thrust::device_pointer_cast(d_diff);
            thrust::inclusive_scan(d_diff_ptr, d_diff_ptr + n, d_diff_ptr);
            record_time(g_scan_time_ns, t4);

//            if (n <= 64) {
//                // d_diff now holds inclusive group IDs (1..G)
//                print_u32_dev(("DEBUG diff (group IDs) (k=" + std::to_string(k) + ")").c_str(), d_diff, n);
//            }

        }

        // 4) assign new ranks => rank[index[i]] = d_diff[i]
        auto t5 = now();
        assign_ranks_kernel<<<gridSize, blockSize>>>(d_index, d_diff, d_rank, n);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        record_time(g_assign_time_ns, t5);

//        if (n <= 64) {
//            print_u32_dev(("DEBUG rank after assign (k=" + std::to_string(k) + ")").c_str(), d_rank, n);
//        }



        // 5) check if all ranks are distinct => if d_diff[n-1] == n
        auto t6 = now();
        uint32_t max_rank;
        CHECK_CUDA_ERROR(cudaMemcpy(&max_rank, d_diff + (n - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost));
        record_time(g_copy_chk_time_ns, t6);

//        if (n <= 64) {
//            std::cout << "DEBUG max_rank=" << max_rank
//                      << "  (k=" << k << ")\n";
//        }


        // all ranks are distinct -> done
        if (max_rank == static_cast<uint32_t>(n)) {
            break;
        }
    }

    // Clean up
    auto t8 = now();
    cudaFree(d_rank);
    cudaFree(d_diff);
    cudaFree(d_keys);
    record_time(g_cleanup_time_ns, t8);

//    if (n <= 64) {
//        // d_index is the SA we’re returning
//        print_u32_dev("DEBUG FINAL SA (d_index)", d_index, n);
//    }
    return d_index; // caller owns d_index and must free
}



// Public wrapper: Returns the host Suffix Array
std::vector<uint32_t> build_suffix_array_prefix_doubling(const std::vector<uint8_t>& s)
{
    std::vector<uint32_t> host_sa;
    if (s.empty()) return host_sa;

    uint32_t* d_sa = build_suffix_array_prefix_doubling_device(s);

    const size_t n     = s.size();
    const size_t bytes = n * sizeof(uint32_t);

    auto t0 = now();

    host_sa.resize(n);
    // Pin the vector's memory to avoid the pageable staging path
    CHECK_CUDA_ERROR(cudaHostRegister(host_sa.data(), bytes, cudaHostRegisterDefault));

    // One big DMA at full speed
    CHECK_CUDA_ERROR(cudaMemcpy(host_sa.data(), d_sa, bytes, cudaMemcpyDeviceToHost));

    record_time(g_copy_back_time_ns, t0);

    CHECK_CUDA_ERROR(cudaHostUnregister(host_sa.data()));
    cudaFree(d_sa);

    return host_sa;
}
