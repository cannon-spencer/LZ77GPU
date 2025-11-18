#pragma once

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        spdlog::error("CUDA error in {} at line {}: {}", __FILE__, __LINE__, cudaGetErrorString(err)); \
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
        spdlog::error("cudaMalloc failed for [{}]: {}", varName, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    return err;
}
