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
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/extrema.h>
#include "libcubwt.cuh"

#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

class MemoryMonitor {
private:
    std::thread monitoring_thread;
    std::atomic<bool> stop_monitoring;
    size_t baselineFreeMem = 0;
    size_t peakUsage = 0;
    constexpr static double BYTES_PER_MB = 1024.0 * 1024.0;

    // Initializes memory tracker by measuring current free memory
    void init() {
        size_t totalMem;
        CHECK_CUDA_ERROR(cudaMemGetInfo(&baselineFreeMem, &totalMem));
        peakUsage = 0;
    }

    // Updates peak memory usage
    void update_peak_usage() {
        size_t currentFree, totalMem;
        CHECK_CUDA_ERROR(cudaMemGetInfo(&currentFree, &totalMem));
        size_t usedNow = (baselineFreeMem > currentFree) ? (baselineFreeMem - currentFree) : 0;
        if (usedNow > peakUsage) {
            peakUsage = usedNow;
        }
    }

    // Function to continuously monitor memory usage
    void monitor() {
        while (!stop_monitoring.load()) {
            update_peak_usage();
            std::this_thread::sleep_for(std::chrono::milliseconds(5)); // Adjust interval as needed
        }
    }

public:
    MemoryMonitor() : stop_monitoring(false) {}

    // Start monitoring
    void start() {
        init();  // Initialize before monitoring
        stop_monitoring.store(false);
        monitoring_thread = std::thread(&MemoryMonitor::monitor, this);
    }

    // Stop monitoring
    void stop() {
        stop_monitoring.store(true);
        if (monitoring_thread.joinable()) {
            monitoring_thread.join();
        }
    }

    // Get peak memory usage during monitoring session
    double get_peak_usage_mb() const {
        return static_cast<double>(peakUsage) / BYTES_PER_MB;
    }

    // Destructor ensures the thread is stopped
    ~MemoryMonitor() {
        stop();
    }
};


// Compares two suffix arrays to determine if they match.
bool compare_SA(const std::vector<size_t>& sa1, const std::vector<size_t>& sa2) {
    if (sa1.size() != sa2.size()) return false;
    for (size_t i = 0; i < sa1.size(); i++) {
        if (sa1[i] != sa2[i]) return false;
    }
    return true;
}

// This struct holds the two ranks (r1, r2) for a suffix and the suffix index. It is sorted by (r1, r2).
struct SuffixKey {
    int r1;         // rank of position i
    int r2;         // rank of position i + k, or 0 if out of range
    size_t idx;     // suffix index

    __host__ __device__
    bool operator<(const SuffixKey& other) const {
        if (r1 != other.r1) return (r1 < other.r1);
        return (r2 < other.r2);
    }

    __host__ __device__
    bool operator==(const SuffixKey& other) const {
        return (r1 == other.r1) && (r2 == other.r2);
    }

    __host__ __device__
    bool operator!=(const SuffixKey& other) const {
        return !(*this == other);
    }
};

/**
 * Kernel to build suffix keys on the device.
 * d_rank contains the integer ranks for each position.
 * d_keys will be populated with (r1, r2, idx).
 * n is the total length, k is the current offset for the second rank.
 */
__global__
void build_keys_kernel(const int* d_rank, SuffixKey* d_keys, size_t n, size_t k){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int r1 = d_rank[i];
        // Use 0 for positions beyond the string to maintain consistent ordering
        int r2 = ((i + k) < n) ? d_rank[i + k] : 0;
        d_keys[i].r1 = r1;
        d_keys[i].r2 = r2;
        d_keys[i].idx = i;
    }
}

/**
 * Kernel to compute difference array.
 * If keys[i] differs from keys[i-1], diff[i] = 1, else 0.
 * The first element is always 1 to mark the start of a new group.
 */
__global__
void compute_diff_kernel(const SuffixKey* d_keys, int* d_diff, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (i == 0) {
            // The first element always starts a new group
            d_diff[0] = 1;
        } else {
            // If the current key differs from the previous one, mark start of new group
            d_diff[i] = (d_keys[i] != d_keys[i-1]) ? 1 : 0;
        }
    }
}

/**
 * Kernel to assign new ranks from the inclusive scan results.
 * After inclusive scan, diff[i] contains the 1-based rank for keys[i].
 */
__global__
void assign_ranks_kernel(const SuffixKey* d_keys, const int* d_new_group_ids, int* d_newRank, size_t n){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        size_t sufIdx = d_keys[i].idx;
        // Assign new rank - groups are identified by their starting position
        d_newRank[sufIdx] = d_new_group_ids[i];
    }
}

/**
 * Builds a suffix array using prefix doubling entirely on the GPU.
 * The input string is in 's'. Initial ranks are set using normalized character values.
 * The final suffix array is returned as a vector<size_t>.
 */
std::vector<size_t> build_suffix_array_prefix_doubling(const std::vector<uint8_t>& s){
    size_t n = s.size();

    // Store rank array in a device vector
    thrust::device_vector<int> d_rank(s.begin(), s.end());

    // Suffix keys used for sorting each iteration
    thrust::device_vector<SuffixKey> d_keys(n);

    // Arrays for calculating new ranks
    thrust::device_vector<int> d_diff(n), d_new_group_ids(n), d_newRank(n);

    // CUDA configuration
    //int blockSize = 256;
    int blockSize = 1024; // testing
    int gridSize = static_cast<int>((n + blockSize - 1) / blockSize);

    // Prefix doubling loop
    for (size_t k = 1; k < n; k *= 2) {
        // Build the (r1, r2, idx) keys
        build_keys_kernel<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(d_rank.data()),
                thrust::raw_pointer_cast(d_keys.data()), n, k);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // Sort the suffix keys by (r1, r2)
        thrust::sort(d_keys.begin(), d_keys.end());

        // Compute differences to identify group boundaries
        compute_diff_kernel<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(d_keys.data()),
                thrust::raw_pointer_cast(d_diff.data()), n);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // Inclusive scan to compute new group IDs
        thrust::inclusive_scan(d_diff.begin(), d_diff.end(), d_new_group_ids.begin());

        // Assign new ranks based on group IDs
        assign_ranks_kernel<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(d_keys.data()),
                thrust::raw_pointer_cast(d_new_group_ids.data()),
                thrust::raw_pointer_cast(d_newRank.data()), n);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // Get the maximum rank
        int max_rank = d_new_group_ids[n - 1];

        // If we have n distinct groups, we're done
        if (max_rank == n) {
            break;
        }

        // Update ranks for next iteration
        thrust::copy(d_newRank.begin(), d_newRank.end(), d_rank.begin());

    }

    // Get the final suffix array from the sorted keys
    std::vector<SuffixKey> hostKeys(n);
    thrust::copy(d_keys.begin(), d_keys.end(), hostKeys.begin());

    std::vector<size_t> sa(n);
    for (size_t i = 0; i < n; i++) {
        sa[i] = hostKeys[i].idx;
    }

    return sa;
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
    std::cout << "Loaded file: " << argv[1] << " (length = " << n << ")\n\n";


    /**
     * PREFIX DOUBLING
     **/

    /// TESTING
    int minGridSize, optBlockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &optBlockSize, build_keys_kernel);
    std::cout << "Optimal Block Size build_keys_kernel: " << optBlockSize << std::endl;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &optBlockSize, compute_diff_kernel);
    std::cout << "Optimal Block Size compute_diff_kernel: " << compute_diff_kernel << std::endl;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &optBlockSize, assign_ranks_kernel);
    std::cout << "Optimal Block Size assign_ranks_kernel: " << optBlockSize << std::endl;

    // init tracker for prefix doubling
    MemoryMonitor prefix_monitor;

    prefix_monitor.start();
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<size_t> sa_pd = build_suffix_array_prefix_doubling(s);
    auto stop = std::chrono::high_resolution_clock::now();
    prefix_monitor.stop();

    auto pd_duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    std::cout << "Prefix Doubling computation time: " << pd_duration << " ms\n";

    // show peak usage in MB for prefix doubling
    std::cout << "Peak GPU memory (prefix doubling): " << prefix_monitor.get_peak_usage_mb() << " MB\n\n";
    
    // Call destructor on memory monitor thread
    prefix_monitor.~MemoryMonitor();


    /**
    * LIBCUBWT TESTING
    **/


    // init SA
    std::vector<uint32_t> SA_cubwt(n);

    void* device_storage = nullptr;
    int64_t err = libcubwt_allocate_device_storage(&device_storage, n);
    if (err != LIBCUBWT_NO_ERROR) {
        std::cerr << "libcubwt_allocate_device_storage error\n";
        return 1;
    }

    // Re-init tracker for libcubwt phase
    MemoryMonitor cubwt_monitor;

    cubwt_monitor.start();
    auto cubwt_start = std::chrono::high_resolution_clock::now();
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

    // Call destructor on memory monitor thread
    cubwt_monitor.~MemoryMonitor();

    /**
     *  FINAL COMPARISON
     * */

    // Convert SA_cubwt => size_t TODO: need to see if i even need to bother doing this
    std::vector<size_t> sa_cubwt(n);
    for (size_t i = 0; i < n; i++) {
        sa_cubwt[i] = static_cast<size_t>(SA_cubwt[i]);
    }

    // Compare results
    bool match = compare_SA(sa_pd, sa_cubwt);
    std::cout << "Checking if both methods produce the same SA...\n";
    if (match) {
        std::cout << "SUCCESS: Both suffix arrays match!\n";
    } else {
        std::cout << "ERROR: The suffix arrays do NOT match.\n";
    }

    return 0;
}
