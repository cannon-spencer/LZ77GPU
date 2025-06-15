#pragma once

#include <iostream>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

inline size_t g_allocated = 0;

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
