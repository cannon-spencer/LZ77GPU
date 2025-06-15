#ifndef LIBCUBWT_DATA_TYPES_CUH
#define LIBCUBWT_DATA_TYPES_CUH

#pragma once
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cstdint>
#include <climits>

#define DEFAULT_BLOCK_SIZE 256
#define DEFAULT_NUM_STREAMS 6
#define DEFAULT_SHARED_MEMORY_SIZE 16384
#define DEFAULT_BATCH_DIV 2
#define DEFAULT_GPU_MEMORY (40ULL * 1024 * 1024 * 1024)

struct Element {
    int64_t value;
    int64_t index;
    __host__ __device__ Element() : value(INT_MAX), index(-1) {}
    __host__ __device__ Element(int v, int i) : value(v), index(i) {}
};

#endif //LIBCUBWT_DATA_TYPES_CUH
