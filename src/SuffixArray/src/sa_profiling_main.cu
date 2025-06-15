#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <chrono>

#include "cuda_utils.cuh"
#include "memory_monitor.cuh"
#include "profiler.cuh"
#include "libcubwt.cuh"
#include "prefix_doubling.cuh"

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
