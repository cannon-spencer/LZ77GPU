#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>
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

/**
 * Check if GPU memory is sufficient for processing large files
 * @param file_size Size of the input file in bytes
 * @param sa_type String indicating SA data type ("uint32_t" or "size_t")
 * @return true if GPU has enough memory, false otherwise
 */
bool checkGPUMemoryForLargeFile(size_t file_size, const std::string& sa_type) {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    
    // memory usage expectation: 3 * SA + 1 * uint8_t + 2 * SA (PSV/NSV) + 20% buffer
    size_t sa_size = (sa_type == "uint32_t") ? sizeof(uint32_t) : sizeof(size_t);
    size_t required_mem = file_size * (3 * sa_size + sizeof(uint8_t) + 2 * sa_size) * 1.2;
    
    std::cout << "GPU Memory Check for " << sa_type << ":" << std::endl;
    std::cout << "  Available: " << free_mem / (1024*1024*1024.0) << " GB" << std::endl;
    std::cout << "  Required: " << required_mem / (1024*1024*1024.0) << " GB" << std::endl;
    
    return free_mem > required_mem;
}


/**
 * Template function for processing LZ77 compression with different SA data types
 * Supports both uint32_t (for files ≤4GB) and size_t (for large files)
 * @param data Input file data as byte vector
 * @param output_prefix Prefix for output files
 */
template<typename SA_t>
void processLZ77(const std::vector<uint8_t>& data, const std::string& output_prefix) {
    size_t length = data.size();
    
    GPUProfiler profiler;
    profiler.start();
    
    if constexpr (std::is_same_v<SA_t, uint32_t>) {
        // uint32_t path: prioritize GPU prefix_doubling for small files
        if (length <= UINT32_MAX) {
            std::cout << "Using 32-bit GPU prefix_doubling..." << std::endl;
            
            try {
                size_t sa_length;
                uint32_t* d_SA = build_suffix_array_prefix_doubling(data, sa_length);
                
                if (d_SA && sa_length == length) {
                    std::cout << "prefix_doubling SA construction completed, SA remains on GPU" << std::endl;
                    profiler.stop("Suffix Array Generation");
                    
                    PipelinePSVNSVProcessor processor;
                    
                    try {
                        //Use GPU SA for zero-copy processing
                        processor.template processFullGPUWithGPUSA<uint32_t>(d_SA, data.data(), length, output_prefix);
                        cudaFree(d_SA);
                        return; // Fastest path, return immediately
                    } catch (...) {
                        cudaFree(d_SA);
                        throw;
                    }
                } else {
                    if (d_SA) cudaFree(d_SA);
                    std::cout << "prefix_doubling failed, falling back to SDSL..." << std::endl;
                }
                
            } catch (const std::exception& e) {
                std::cout << "prefix_doubling exception: " << e.what() 
                         << ", falling back to SDSL..." << std::endl;
            }
        }
        
    } else if constexpr (std::is_same_v<SA_t, size_t>) {
        // New feature: size_t path also attempts GPU prefix_doubling for large files
        std::cout << "Using 64-bit GPU prefix_doubling for large file..." << std::endl;
        
        // Check if GPU memory is sufficient
        if (checkGPUMemoryForLargeFile(length, "size_t")) {
            try {
                size_t sa_length;
                size_t* d_SA = build_suffix_array_prefix_doubling_64(data, sa_length);
                
                if (d_SA && sa_length == length) {
                    std::cout << "64-bit prefix_doubling SA construction completed, SA remains on GPU" << std::endl;
                    profiler.stop("Suffix Array Generation");
                    
                    PipelinePSVNSVProcessor processor;
                    
                    try {
                        // Use GPU SA for zero-copy processing (size_t version)
                        processor.template processFullGPUWithGPUSA<size_t>(d_SA, data.data(), length, output_prefix);
                        cudaFree(d_SA);
                        return; //Large file GPU path, return immediately
                    } catch (...) {
                        cudaFree(d_SA);
                        throw;
                    }
                } else {
                    if (d_SA) cudaFree(d_SA);
                    std::cout << "64-bit prefix_doubling failed, falling back to SDSL..." << std::endl;
                }
                
            } catch (const std::exception& e) {
                std::cout << "64-bit prefix_doubling exception: " << e.what() 
                         << ", falling back to SDSL..." << std::endl;
            }
        } else {
            std::cout << "Insufficient GPU memory for 64-bit prefix_doubling, using SDSL..." << std::endl;
        }
    }

    // Fallback: Use SDSL (for large files or when prefix_doubling fails)
    std::vector<SA_t> SA(length);
    
    if constexpr (std::is_same_v<SA_t, uint32_t>) {
        try {
            sdsl::int_vector<32> sdsl_sa(length);
            sdsl::algorithm::calculate_sa(static_cast<const unsigned char *>(data.data()), length, sdsl_sa);
            
            // Convert from SDSL to uint32_t
            for (size_t i = 0; i < length; ++i) {
                SA[i] = static_cast<uint32_t>(sdsl_sa[i]);
            }
            std::cout << "SDSL SA construction finished (uint32_t)" << std::endl;
            
        } catch (const std::exception& sdsl_e) {
            std::cerr << "Failed to construct suffix array using SDSL: " << sdsl_e.what() << std::endl;
            throw;
        }
    } else {
        // Use SDSL for size_t (large files)
        try {
            std::cout << "Using SDSL for SA construction (size_t)" << std::endl;
            sdsl::int_vector<sizeof(size_t) * 8> sdsl_sa(length);
            sdsl::algorithm::calculate_sa(static_cast<const unsigned char *>(data.data()), length, sdsl_sa);
            std::memcpy(SA.data(), sdsl_sa.data(), length * sizeof(size_t));
            std::cout << "SDSL SA construction finished (size_t)" << std::endl;
            
        } catch (const std::exception& e) {
            std::cerr << "Failed to construct suffix array using SDSL: " << e.what() << std::endl;
            throw;
        }
    }
    
    profiler.stop("Suffix Array Generation");
    std::cout << "Before processor initialized" << std::endl;
    
    // Create templated processor
    PipelinePSVNSVProcessor processor;
    std::cout << "After processor initialized" << std::endl;
    
    // Process with appropriate type using regular process function
    processor.template process<SA_t>(SA.data(), data.data(), length, output_prefix);

}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_prefix>" << std::endl;
        return 1;
    }

    std::string input_file = argv[1];
    std::string output_prefix = argv[2];

    std::ifstream file(input_file, std::ios::binary);

    if (!file) {
        std::cerr << "Cannot open file: " << input_file << std::endl;
        return 1;
    }

    // Read the file into a vector of uint8_t
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    if (data.empty()) {
        std::cerr << "File is empty or could not be read correctly." << std::endl;
        return 1;
    }

    // Add null terminator for suffix array construction
    data.push_back(0);
    size_t length = data.size();
    
    std::cout << "Input file size: " << length << " bytes" << std::endl;

    try {
        
        //GPU memory check and processing path selection
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        std::cout << "GPU: " << free_mem / (1024*1024*1024.0) << " GB free / " 
                  << total_mem / (1024*1024*1024.0) << " GB total" << std::endl;

        // Choose SA type based on file size
        if (length <= UINT32_MAX) {
            std::cout << "File size fits in uint32_t, using optimized 32-bit processing" << std::endl;
            processLZ77<uint32_t>(data, output_prefix);
        } else {
            std::cout << "File size requires size_t, using 64-bit processing" << std::endl;
            processLZ77<size_t>(data, output_prefix);
        }
        
        std::cout << "LZ77 compression completed successfully" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error occurred: " << e.what() << std::endl;
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        }
        return 1;
    }

    return 0;
}


