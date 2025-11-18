#include <fstream>
#include <vector>
#include <cstdint>
#include <chrono>
#include <ctime>
#include <cstdlib>
#include <iomanip>

#include "cuda_utils.cuh"
#include "memory_monitor.cuh"
#include "profiler.cuh"
#include "libcubwt.cuh"
#include "prefix_doubling.cuh"
#include "logger.cuh"

void print_profiling_summary() {
    LOG_INFO("==== Profiling Summary ====");
    LOG_INFO("Initialization time:      {:.2f} ms", g_alloc_time_ns / 1e6);
    LOG_INFO("Total Sort time:          {:.2f} ms", g_sort_time_ns / 1e6);
    LOG_INFO("Compute Diff Kernel time: {:.2f} ms", g_kernel_diff_time_ns / 1e6);
    LOG_INFO("Inclusive Scan time:      {:.2f} ms", g_scan_time_ns / 1e6);
    LOG_INFO("Assign Ranks Kernel time: {:.2f} ms", g_kernel_assign_time_ns / 1e6);
    LOG_INFO("Max Rank Host Copy time:  {:.2f} ms", g_copy_time_ns / 1e6);
    LOG_INFO("Deallocation time:        {:.2f} ms", g_cleanup_time_ns / 1e6);
    LOG_INFO("===========================");
}


// Optionally print suffix array output for debugging
template <typename SA_t>
void dump_sa(const std::vector<SA_t>& sa,
             const char* tag,
             size_t max_lines = SIZE_MAX)    // PASS n to print all
{
    LOG_INFO("\n--- {} (size={}) ---", tag, sa.size());
    size_t shown = 0;
    for (size_t i = 0; i < sa.size(); ++i) {
        if (shown++ == max_lines) {          // stop after max_lines
            LOG_INFO("  ... (truncated)");
            break;
        }
        LOG_INFO("{:6}: {}", i, sa[i]);
    }
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
    // Initialize logger
    lz77gpu::init_logger();

    /**
     *  READ THE INPUT FILE
     * */

    if (argc < 2) {
        LOG_ERROR("Usage: {} <input_file>", argv[0]);
        return 1;
    }

    std::ifstream fin(argv[1], std::ios::binary);
    if (!fin.is_open()) {
        LOG_ERROR("Cannot open file: {}", argv[1]);
        return 1;
    }
    std::vector<uint8_t> s((std::istreambuf_iterator<char>(fin)), {});
    fin.close();

    if (s.empty()) {
        LOG_ERROR("File is empty.");
        return 1;
    }

    size_t n = s.size();
    LOG_INFO("Loaded file: {} (length = {} MB)\n", argv[1], n / (1024 * 1024));

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
        LOG_ERROR("libcubwt_allocate_device_storage error");
        return 1;
    }


    // Compute Suffix Array
    err = libcubwt_sa(device_storage, reinterpret_cast<const uint8_t*>(s.data()), SA_cubwt.data(), n);
    auto cubwt_stop = std::chrono::high_resolution_clock::now();
    cubwt_monitor.stop();

    if (err != LIBCUBWT_NO_ERROR) {
        LOG_ERROR("libcubwt_sa error");
        return 1;
    }

    // Free device storage used by libcubwt
    libcubwt_free_device_storage(device_storage);

    auto cubwt_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cubwt_stop - cubwt_start).count();
    LOG_INFO("libcubwt SA computation time: {} ms", cubwt_duration);

    // final peak usage for libcubwt
    LOG_INFO("Peak GPU memory (libcubwt): {:.2f} MB\n", cubwt_monitor.get_peak_usage_mb());

    /**
     * PREFIX DOUBLING
     **/

    // init tracker for prefix doubling
    MemoryMonitor prefix_monitor;

    prefix_monitor.start();
    auto start = std::chrono::high_resolution_clock::now();
    size_t sa_length;
    uint32_t* d_SA_pd = build_suffix_array_prefix_doubling(s, sa_length);
    auto stop = std::chrono::high_resolution_clock::now();
    prefix_monitor.stop();

    auto pd_duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    LOG_INFO("Prefix Doubling computation time: {} ms", pd_duration);

    // show peak usage in MB for prefix doubling
    LOG_INFO("Peak GPU memory (prefix doubling): {:.2f} MB\n", prefix_monitor.get_peak_usage_mb());

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
    std::vector<uint32_t> SA_pd(n);
    cudaError_t cuda_err = cudaMemcpy(SA_pd.data(), d_SA_pd, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (cuda_err != cudaSuccess) {
        LOG_ERROR("Failed to copy SA from GPU to host: {}", cudaGetErrorString(cuda_err));
        cudaFree(d_SA_pd);
        return 1;
    }

    // Clean up GPU memory after copying to host
    cudaFree(d_SA_pd);

    bool match = compare_SA(SA_pd, SA_cubwt);
    //match = match && compare_SA(SA_cubwt, SA_sdsl);
    LOG_INFO("Checking if both methods produce the same SA...");
    if (match) {
        LOG_INFO("SUCCESS: Both suffix arrays match!");
    } else {
        LOG_ERROR("ERROR: The suffix arrays do NOT match.");

        // debug print the arrays
        dump_sa(SA_pd,    "Prefix-doubling SA", SA_pd.size());
        dump_sa(SA_cubwt, "libcubwt SA",        SA_cubwt.size());
    }

    return 0;
}
