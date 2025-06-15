#include <iostream>
#include <algorithm>
#include <thread>
#include <atomic>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <chrono>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/scan.h>
#include <sdsl/suffix_arrays.hpp>

#include "libcubwt.cuh"

/**
 * CUDA HELPER FUNCTIONS
 */
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

static size_t g_allocated = 0;

template <typename T>
cudaError_t myCudaMalloc(T** ptr, size_t size, const char* varName) {
    cudaError_t err = cudaMalloc(ptr, size);
    if (err == cudaSuccess) {
        g_allocated += size;
    } else {
        std::cerr << "cudaMalloc failed for [" << varName << "]: "
                  << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
    return err;
}


/**
 * CUDA SA PROFILING FUNCTIONS
 */

uint64_t g_alloc_time_ns = 0;
uint64_t g_sort_time_ns = 0;
uint64_t g_kernel_diff_time_ns = 0;
uint64_t g_scan_time_ns = 0;
uint64_t g_kernel_assign_time_ns = 0;
uint64_t g_copy_time_ns = 0;
uint64_t g_cleanup_time_ns = 0;

inline auto now() { return std::chrono::high_resolution_clock::now(); }

inline void record_time(uint64_t& accumulator, const std::chrono::high_resolution_clock::time_point& start) {
    auto end = std::chrono::high_resolution_clock::now();
    accumulator += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

void print_profiling_summary() {
    std::cout << "==== Profiling Summary ====\n";
    std::cout << "Initialization time:      " << g_alloc_time_ns / 1e6 << " ms\n";
    std::cout << "Total Sort time:          " << g_sort_time_ns / 1e6 << " ms\n";
    std::cout << "Compute Diff Kernel time: " << g_kernel_diff_time_ns / 1e6 << " ms\n";
    std::cout << "Inclusive Scan time:      " << g_scan_time_ns / 1e6 << " ms\n";
    std::cout << "Assign Ranks Kernel time: " << g_kernel_assign_time_ns / 1e6 << " ms\n";
    std::cout << "Max Rank Host Copy time:  " << g_copy_time_ns / 1e6 << " ms\n";
    std::cout << "Deallocation time:        " << g_cleanup_time_ns / 1e6 << " ms\n";
    std::cout << "===========================\n";
}

/**
 * MEMORY MONITOR FOR TRACKING PEAK GPU MEMORY USAGE
 */
class MemoryMonitor {
private:
    std::thread monitoring_thread;
    std::atomic<bool> stop_monitoring;
    size_t baselineFreeMem = 0;
    size_t peakUsage = 0;
    constexpr static double BYTES_PER_MB = 1024.0 * 1024.0;

    void init() {
        size_t totalMem;
        CHECK_CUDA_ERROR(cudaMemGetInfo(&baselineFreeMem, &totalMem));
        peakUsage = 0;
    }

    void update_peak_usage() {
        size_t currentFree, totalMem;
        CHECK_CUDA_ERROR(cudaMemGetInfo(&currentFree, &totalMem));
        size_t usedNow = (baselineFreeMem > currentFree) ? (baselineFreeMem - currentFree) : 0;
        if (usedNow > peakUsage) {
            peakUsage = usedNow;
        }
    }

    void monitor() {
        while (!stop_monitoring.load()) {
            update_peak_usage();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

public:
    MemoryMonitor() : stop_monitoring(false) {}

    void start() {
        init();
        stop_monitoring.store(false);
        monitoring_thread = std::thread(&MemoryMonitor::monitor, this);
    }

    void stop() {
        stop_monitoring.store(true);
        if (monitoring_thread.joinable()) {
            monitoring_thread.join();
        }
    }

    double get_peak_usage_mb() const {
        return static_cast<double>(peakUsage) / BYTES_PER_MB;
    }

    ~MemoryMonitor() {
        stop();
    }
};

// Compare two SAs
template <typename T, typename U>
bool compare_SA(const std::vector<T>& sa1, const std::vector<U>& sa2) {
    if (sa1.size() != sa2.size()) {
        return false;
    }
    for (size_t i = 0; i < sa1.size(); i++) {
        if (sa1[i] != sa2[i]) {
            return false;
        }
    }
    return true;
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


int main(int argc, char** argv){
    /**
     *  READ THE INPUT FILE
     * */

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>\n";
        return 1;
    }

    std::ifstream fin(argv[1], std::ios::binary);
    if (!fin.is_open()) {
        std::cerr << "Cannot open file: " << argv[1] << "\n";
        return 1;
    }
    std::vector<uint8_t> s((std::istreambuf_iterator<char>(fin)), {});
    fin.close();

    if (s.empty()) {
        std::cerr << "File is empty.\n";
        return 1;
    }

    size_t n = s.size();
    std::cout << "Loaded file: " << argv[1] << " (length = " << n / (1024 * 1024) << " MB)\n\n";

    /**
    * LIBCUBWT TESTING
    **/


    // Re-init tracker for libcubwt phase
    MemoryMonitor cubwt_monitor;

    cubwt_monitor.start();
    auto cubwt_start = std::chrono::high_resolution_clock::now();

    // init SA
    std::vector<uint32_t> SA_cubwt(n);

    // Allocate Memory for libcubwt
    void* device_storage = nullptr;
    int64_t err = libcubwt_allocate_device_storage(&device_storage, n);
    if (err != LIBCUBWT_NO_ERROR) {
        std::cerr << "libcubwt_allocate_device_storage error\n";
        return 1;
    }


    // Compute Suffix Array
    err = libcubwt_sa(device_storage, reinterpret_cast<const uint8_t*>(s.data()), SA_cubwt.data(), n);
    auto cubwt_stop = std::chrono::high_resolution_clock::now();
    cubwt_monitor.stop();

    if (err != LIBCUBWT_NO_ERROR) {
        std::cerr << "libcubwt_sa error\n";
        return 1;
    }

    // Free device storage used by libcubwt
    libcubwt_free_device_storage(device_storage);

    auto cubwt_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cubwt_stop - cubwt_start).count();
    std::cout << "libcubwt SA computation time: " << cubwt_duration << " ms\n";

    // final peak usage for libcubwt
    std::cout << "Peak GPU memory (libcubwt): " << cubwt_monitor.get_peak_usage_mb() << " MB\n\n";


    /**
     * PREFIX DOUBLING
     **/


    // init tracker for prefix doubling
    MemoryMonitor prefix_monitor;

    prefix_monitor.start();
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<uint32_t> SA_pd = build_suffix_array_prefix_doubling(s);
    auto stop = std::chrono::high_resolution_clock::now();
    prefix_monitor.stop();

    auto pd_duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    std::cout << "Prefix Doubling computation time: " << pd_duration << " ms\n";

    // show peak usage in MB for prefix doubling
    std::cout << "Peak GPU memory (prefix doubling): " << prefix_monitor.get_peak_usage_mb() << " MB\n\n";

    // output the profiler for sections of the prefix doubling
    print_profiling_summary();

    /**
     * SDSL (CPU) Suffix Array
     **/

    /*
    // init tracker for SDSL version
    auto sdsl_start = std::chrono::high_resolution_clock::now();

    std::vector<size_t> SA_sdsl(n);

    try {
        sdsl::int_vector<sizeof(size_t) * 8> sdsl_sa(n);
        sdsl::algorithm::calculate_sa(static_cast<const unsigned char*>(s.data()), n, sdsl_sa);
        std::memcpy(SA_sdsl.data(), sdsl_sa.data(), n * sizeof(size_t));
    }
    catch (const std::exception& e) {
        std::cerr << "Failed to construct suffix array using SDSL: " << e.what() << std::endl;
        return 1;
    }

    auto sdsl_stop = std::chrono::high_resolution_clock::now();

    auto sdsl_duration = std::chrono::duration_cast<std::chrono::milliseconds>(sdsl_stop - sdsl_start).count();
    std::cout << "SDSL SA computation time: " << sdsl_duration << " ms\n";
    */


    /**
     *  FINAL COMPARISON
     * */

    // Compare results
    bool match = compare_SA(SA_pd, SA_cubwt);
    //match = match && compare_SA(SA_cubwt, SA_sdsl);
    std::cout << "Checking if both methods produce the same SA...\n";
    if (match) {
        std::cout << "SUCCESS: Both suffix arrays match!\n";
    } else {
        std::cout << "ERROR: The suffix arrays do NOT match.\n";
    }

    return 0;
}
