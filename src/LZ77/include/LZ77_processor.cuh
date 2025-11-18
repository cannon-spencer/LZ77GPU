#ifndef LZ77_PROCESSOR_CUH
#define LZ77_PROCESSOR_CUH

#include <vector>
#include <string>
#include <cstdint>
#include <type_traits>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

constexpr size_t DEFAULT_BLOCK_SIZE = 1024;
constexpr float MEMORY_RESERVE_RATIO = 0.9f;

// Helper template functions for handling SA_t types
template<typename SA_t>
constexpr SA_t get_max_value() {
    if constexpr (std::is_same_v<SA_t, uint32_t>) {
        return UINT32_MAX;
    } else {
        return SIZE_MAX;
    }
}

template<typename SA_t>
constexpr bool is_invalid_value(SA_t value) {
    return value == get_max_value<SA_t>();
}

// CUDA kernels
template<typename SA_t>
__global__ void computePSVNSVKernel(
    const SA_t* __restrict__ input,
    SA_t* __restrict__ psv_output,
    SA_t* __restrict__ nsv_output,
    SA_t* __restrict__ block_min_output,
    const size_t length
);

template<typename SA_t>
__global__ void processPSVNSVBoundariesKernel(
    const SA_t* __restrict__ sa_array,
    SA_t* __restrict__ psv_sa_order,
    SA_t* __restrict__ nsv_sa_order,
    const SA_t* __restrict__ block_min_output,
    const size_t length,
    const size_t block_size
);

template<typename SA_t>
__global__ void dualScatterKernel(
    const SA_t* __restrict__ psv_values,
    const SA_t* __restrict__ nsv_values,
    const SA_t* __restrict__ indices,
    SA_t* __restrict__ psv_output,
    SA_t* __restrict__ nsv_output,
    size_t chunk_size
);

class GPUProfiler {
public:
    GPUProfiler();
    ~GPUProfiler();
    void start();
    float stop(const char* operation_name);

private:
    cudaEvent_t start_event, stop_event;
};

// Chunk metadata for streaming mode
template<typename SA_t>
struct ChunkMetadata {
    size_t start_offset;
    size_t chunk_size;
    SA_t min_value;
    SA_t max_value;
    std::vector<size_t> psv_unfound_indices;  // Global indices where PSV not found
    std::vector<size_t> nsv_unfound_indices;  // Global indices where NSV not found
};

class PipelinePSVNSVProcessor {
public:
    PipelinePSVNSVProcessor();

    // Path 1: Full GPU processing with GPU SA (zero-copy, fastest)
    template<typename SA_t>
    void processFullGPUWithGPUSA(SA_t* d_sa_array, const uint8_t* data, size_t length, const std::string& output_prefix);

    // Path 4: Stream processing with CPU SA (memory-limited)
    template<typename SA_t>
    void processWithStreams(std::vector<SA_t>& sa_array, const uint8_t* data, size_t length, const std::string& output_prefix);

private:
    GPUProfiler profiler;
    size_t available_memory;

    void calculateAvailableMemory();

    template<typename SA_t>
    std::pair<std::pair<size_t, size_t>, size_t> LZFactor(const uint8_t *data, size_t i, SA_t psv, SA_t nsv, size_t n);

    template<typename SA_t>
    void ComputeLZ77(const uint8_t *data, SA_t *d_psv_text, SA_t *d_nsv_text, size_t n, std::string file_name);

    // Optimized targeted search with block_min pruning
    template<typename SA_t>
    void resolvePSVWithPruning(const SA_t* sa_array, SA_t* psv,
                               const std::vector<size_t>& unfound_indices,
                               const std::vector<SA_t>& block_mins,
                               size_t length, size_t block_size);

    template<typename SA_t>
    void resolveNSVWithPruning(const SA_t* sa_array, SA_t* nsv,
                               const std::vector<size_t>& unfound_indices,
                               const std::vector<SA_t>& block_mins,
                               size_t length, size_t block_size);

    // GPU streaming text-order conversion for memory-constrained scenarios
    template<typename SA_t>
    void convertToTextOrderGPUStreaming(const SA_t* sa_array, SA_t* psv, SA_t* nsv, size_t length);
};

#endif // LZ77_PROCESSOR_CUH