#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>
#include <vector>
#include <cuda_runtime.h>

#include "prefix_doubling.cuh"

// Helper function to get SA from GPU and clean up memory
std::vector<uint32_t> get_sa_from_gpu(const std::vector<uint8_t>& bytes) {
    size_t sa_length;
    uint32_t* d_sa = build_suffix_array_prefix_doubling(bytes, sa_length);
    
    if (!d_sa || sa_length != bytes.size()) {
        if (d_sa) cudaFree(d_sa);
        throw std::runtime_error("Failed to build suffix array");
    }
    
    // Copy from GPU to host
    std::vector<uint32_t> sa(sa_length);
    cudaError_t err = cudaMemcpy(sa.data(), d_sa, sa_length * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    // Clean up GPU memory
    cudaFree(d_sa);
    
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy SA from GPU: " + std::string(cudaGetErrorString(err)));
    }
    
    return sa;
}

TEST_CASE("Suffix Array: MISSISSIPPI Example") {
    std::string input = "MISSISSIPPIMISSISSIPPI";
    std::vector<uint8_t> bytes(input.begin(), input.end());
    std::vector<uint32_t> expected_sa = {
        21, 10, 18, 7, 15, 4, 12, 1, 11, 0, 20, 9, 19, 8, 17, 6, 14, 3, 16, 5, 13, 2
    };

    SECTION("Output size is correct") {
        auto sa = get_sa_from_gpu(bytes);
        REQUIRE(sa.size() == bytes.size());
    }

    SECTION("Matches reference suffix array") {
        auto sa = get_sa_from_gpu(bytes);
        REQUIRE(sa == expected_sa);
    }

    SECTION("Suffixes are in lexicographic order") {
        auto sa = get_sa_from_gpu(bytes);
        std::vector<std::string> suffixes;
        for (uint32_t i : sa)
            suffixes.emplace_back(input.substr(i));
        for (size_t i = 1; i < suffixes.size(); ++i)
            REQUIRE(suffixes[i-1] <= suffixes[i]);
    }

    SECTION("ISA is the true inverse") {
        auto sa = get_sa_from_gpu(bytes);
        std::vector<uint32_t> isa(sa.size());
        for (size_t i = 0; i < sa.size(); ++i)
            isa[sa[i]] = i;
        for (size_t i = 0; i < sa.size(); ++i)
            REQUIRE(sa[isa[i]] == i);
    }
}

TEST_CASE("Suffix Array: Error Handling") {
    SECTION("Empty input") {
        std::vector<uint8_t> empty_bytes;
        size_t sa_length;
        uint32_t* d_sa = build_suffix_array_prefix_doubling(empty_bytes, sa_length);
        
        REQUIRE(d_sa == nullptr);
        REQUIRE(sa_length == 0);
    }
    
    SECTION("Single character") {
        std::vector<uint8_t> single_char = {'A'};
        auto sa = get_sa_from_gpu(single_char);
        
        REQUIRE(sa.size() == 1);
        REQUIRE(sa[0] == 0);
    }
    
    SECTION("GPU memory management") {
        std::vector<uint8_t> test_data = {'A', 'B', 'C'};
        
        // Test that we can call the function multiple times without memory leaks
        for (int i = 0; i < 5; ++i) {
            auto sa = get_sa_from_gpu(test_data);
            REQUIRE(sa.size() == 3);
        }
        
        // Test direct GPU pointer management
        size_t sa_length;
        uint32_t* d_sa = build_suffix_array_prefix_doubling(test_data, sa_length);
        
        REQUIRE(d_sa != nullptr);
        REQUIRE(sa_length == 3);
        
        // Verify we can access GPU memory
        std::vector<uint32_t> sa(sa_length);
        cudaError_t err = cudaMemcpy(sa.data(), d_sa, sa_length * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        REQUIRE(err == cudaSuccess);
        
        // Clean up
        cudaFree(d_sa);
        
        REQUIRE(sa.size() == 3);
    }
}

TEST_CASE("Suffix Array: Performance and Correctness") {
    SECTION("Larger input") {
        std::string pattern = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        std::string large_input;
        for (int i = 0; i < 100; ++i) {
            large_input += pattern;
        }
        
        std::vector<uint8_t> bytes(large_input.begin(), large_input.end());
        auto sa = get_sa_from_gpu(bytes);
        
        REQUIRE(sa.size() == bytes.size());
        
        // Verify lexicographic order
        for (size_t i = 1; i < sa.size(); ++i) {
            std::string suffix1(large_input.substr(sa[i-1]));
            std::string suffix2(large_input.substr(sa[i]));
            REQUIRE(suffix1 <= suffix2);
        }
    }
    
    SECTION("Repeated patterns") {
        std::string input = "AAAAAAAAAA";
        std::vector<uint8_t> bytes(input.begin(), input.end());
        auto sa = get_sa_from_gpu(bytes);
        
        REQUIRE(sa.size() == bytes.size());
        
        // For repeated characters, SA should be in descending order
        for (size_t i = 1; i < sa.size(); ++i) {
            REQUIRE(sa[i-1] > sa[i]);
        }
    }
}