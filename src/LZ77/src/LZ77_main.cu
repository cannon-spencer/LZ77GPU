#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <fstream>
#include <chrono>
#include <sdsl/suffix_arrays.hpp>
#include <iomanip>
#include <atomic>

#include "prefix_doubling.cuh"
#include "LZ77_processor.cuh"
#include "logger.cuh"

// libsais for fast SA construction
#include <libsais.h>
#include <libsais64.h>



/**
 * Calculate required GPU memory for full GPU processing
 * @param length Number of elements (text length)
 * @param sa_size Size of SA element (sizeof(uint32_t) or sizeof(size_t))
 * @return Required memory in bytes
 */
size_t calculateRequiredMemory(size_t length, size_t sa_size) {
    // Memory usage:
    // - SA on GPU: 1 * length * sa_size (will be constructed)
    // - PSV/NSV output: 2 * length * sa_size
    // - Temp buffer for scatter: 1 * length * sa_size
    // - Block mins: ~num_blocks * sa_size (~0.001 * length * sa_size, negligible)
    // Total: 4 * length * sa_size
    // 20% buffer for safety
    return length * sa_size * 4 * 1.2;
}

/**
 * Build suffix array on GPU using prefix_doubling
 * @param data Input data
 * @return Device pointer to SA, or nullptr on failure
 */
template<typename SA_t>
SA_t* build_SA_on_GPU(const std::vector<uint8_t>& data) {
    size_t sa_length;
    SA_t* d_SA = nullptr;

    if constexpr (std::is_same_v<SA_t, uint32_t>) {
        d_SA = build_suffix_array_prefix_doubling(data, sa_length);
    } else if constexpr (std::is_same_v<SA_t, size_t>) {
        d_SA = build_suffix_array_prefix_doubling_64(data, sa_length);
    }

    if (!d_SA || sa_length != data.size()) {
        if (d_SA) cudaFree(d_SA);
        throw std::runtime_error("Failed to construct suffix array on GPU");
    }

    return d_SA;
}

/**
 * Build suffix array on CPU using libsais (fastest CPU SA construction)
 * @param data Input data
 * @return Host vector containing SA
 */
template<typename SA_t>
std::vector<SA_t> build_SA_on_CPU(const std::vector<uint8_t>& data) {
    size_t length = data.size();
    std::vector<SA_t> SA(length);

    if constexpr (std::is_same_v<SA_t, uint32_t>) {
        // Use libsais for 32-bit (files <= 2GB)
        std::vector<int32_t> sa_tmp(length);

        // Use OpenMP version if available for parallel construction
        #if defined(_OPENMP)
        int threads = omp_get_max_threads();
        libsais_omp(data.data(), sa_tmp.data(), static_cast<int32_t>(length), 0, nullptr, threads);
        #else
        libsais(data.data(), sa_tmp.data(), static_cast<int32_t>(length), 0, nullptr);
        #endif

        // Parallel copy int32_t -> uint32_t
        #pragma omp parallel for
        for (size_t i = 0; i < length; ++i) {
            SA[i] = static_cast<uint32_t>(sa_tmp[i]);
        }

    } else if constexpr (std::is_same_v<SA_t, size_t>) {
        // Use libsais64 for 64-bit (files > 2GB)
        std::vector<int64_t> sa_tmp(length);

        // Use OpenMP version if available for parallel construction
        #if defined(_OPENMP)
        int threads = omp_get_max_threads();
        libsais64_omp(data.data(), sa_tmp.data(), static_cast<int64_t>(length), 0, nullptr, threads);
        #else
        libsais64(data.data(), sa_tmp.data(), static_cast<int64_t>(length), 0, nullptr);
        #endif

        // Parallel copy int64_t -> size_t
        #pragma omp parallel for
        for (size_t i = 0; i < length; ++i) {
            SA[i] = static_cast<size_t>(sa_tmp[i]);
        }
    }

    return SA;
}

/**
 * Template function for processing LZ77 compression with different SA data types
 * Strategy: Check memory first, then choose SA construction method and processing path
 * @param data Input file data as byte vector
 * @param output_prefix Prefix for output files
 */
template<typename SA_t>
void processLZ77(const std::vector<uint8_t>& data, const std::string& output_prefix) {
    size_t length = data.size();

    // Step 1: Check GPU memory to decide which path to take
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    size_t required_mem = calculateRequiredMemory(length, sizeof(SA_t));

    LOG_INFO("\n=== Memory Check ===");
    LOG_INFO("GPU Free Memory: {:.3f} GB", free_mem / (1024*1024*1024.0));
    LOG_INFO("Required Memory: {:.3f} GB", required_mem / (1024*1024*1024.0));

    GPUProfiler profiler;
    PipelinePSVNSVProcessor processor;

    if (free_mem > required_mem) {
        // Path 1: Full GPU processing (zero-copy, fastest)
        LOG_INFO("\n=== Path 1: Full GPU Mode ===");
        LOG_INFO("Building SA on GPU...");

        profiler.start();
        SA_t* d_SA = build_SA_on_GPU<SA_t>(data);
        profiler.stop("GPU SA Construction");

        LOG_INFO("SA construction completed, SA remains on GPU (zero-copy)");

        try {
            processor.template processFullGPUWithGPUSA<SA_t>(d_SA, data.data(), length, output_prefix);
            cudaFree(d_SA);
        } catch (...) {
            cudaFree(d_SA);
            throw;
        }

    } else {
        // Path 4: Stream processing (memory-limited)
        LOG_INFO("\n=== Path 4: Stream Mode ===");
        LOG_INFO("Building SA on CPU (SDSL)...");

        profiler.start();
        std::vector<SA_t> h_SA = build_SA_on_CPU<SA_t>(data);
        profiler.stop("CPU SA Construction");

        LOG_INFO("SA construction completed on CPU");

        processor.template processWithStreams<SA_t>(h_SA, data.data(), length, output_prefix);
    }
}

int main(int argc, char **argv) {
    // Initialize logger
    lz77gpu::init_logger();

    if (argc < 3) {
        LOG_ERROR("Usage: {} <input_file> <output_prefix> [options]", argv[0]);
        LOG_ERROR("Options:");
        LOG_ERROR("  --force-size-t    Force size_t SA type (test 64-bit path)");
        return 1;
    }

    std::string input_file = argv[1];
    std::string output_prefix = argv[2];

    // Parse optional flags
    bool force_size_t = false;

    for (int i = 3; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--force-size-t") {
            force_size_t = true;
        } else {
            LOG_ERROR("Unknown option: {}", arg);
            return 1;
        }
    }

    std::ifstream file(input_file, std::ios::binary);

    if (!file) {
        LOG_ERROR("Cannot open file: {}", input_file);
        return 1;
    }

    // Read the file into a vector of uint8_t
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    if (data.empty()) {
        LOG_ERROR("File is empty or could not be read correctly.");
        return 1;
    }

    // Add null terminator for suffix array construction
    data.push_back(0);
    size_t length = data.size();

    LOG_INFO("Input file size: {} bytes", length);

    try {
        // Display GPU info
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        LOG_INFO("GPU: {:.3f} GB free / {:.3f} GB total",
                 free_mem / (1024*1024*1024.0), total_mem / (1024*1024*1024.0));

        if (force_size_t) {
            LOG_INFO("SA Type: size_t (forced, 64-bit)");
        }

        // Choose SA type based on file size or forced option
        if (force_size_t || length > UINT32_MAX) {
            if (length <= UINT32_MAX) {
                LOG_INFO("Note: File size fits in uint32_t but using size_t for testing");
            } else {
                LOG_INFO("File size requires size_t, using 64-bit processing");
            }
            processLZ77<size_t>(data, output_prefix);
        } else {
            LOG_INFO("File size fits in uint32_t, using optimized 32-bit processing");
            processLZ77<uint32_t>(data, output_prefix);
        }

        LOG_INFO("\nLZ77 compression completed successfully");

    } catch (const std::exception& e) {
        LOG_ERROR("Error occurred: {}", e.what());
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            LOG_ERROR("CUDA error: {}", cudaGetErrorString(err));
        }
        return 1;
    }

    return 0;
}


