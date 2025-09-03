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

// Template function for processing LZ77 with different SA types
template<typename SA_t>
void processLZ77(const std::vector<uint8_t>& data, const std::string& output_prefix) {
    size_t length = data.size();
    
    GPUProfiler profiler;
    profiler.start();
    
    if constexpr (std::is_same_v<SA_t, uint32_t>) {
        // Use prefix_doubling for uint32_t (small files)
        try {
            std::cout << "Using prefix_doubling for SA construction (uint32_t)" << std::endl;
            size_t sa_length;
            uint32_t* d_SA = build_suffix_array_prefix_doubling(data, sa_length);
            
            // Verify the returned suffix array size
            if (d_SA && sa_length == length) {
                std::cout << "prefix_doubling SA construction completed, SA remains on GPU" << std::endl;
                profiler.stop("Suffix Array Generation");
                
                PipelinePSVNSVProcessor processor;
                
                try {
                    // Use GPU SA directly for full GPU processing (zero-copy optimization)
                    std::cout << "Using full GPU processing with GPU SA (zero-copy)" << std::endl;
                    processor.template processFullGPUWithGPUSA<uint32_t>(d_SA, data.data(), length, output_prefix);
                } catch (...) {
                    // Ensure GPU memory cleanup on exception
                    cudaFree(d_SA);
                    throw;
                }
                
                // Clean up GPU SA memory
                cudaFree(d_SA);
                return; // Early return, skip SDSL fallback
            } else {
                if (d_SA) cudaFree(d_SA);
                throw std::runtime_error("prefix_doubling failed");
            }
            
        } catch (const std::exception& e) {
            // Fallback to SDSL for uint32_t
            std::cout << "prefix_doubling failed (" << e.what() << "), switching to SDSL" << std::endl;
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


