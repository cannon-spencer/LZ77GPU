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
__global__ void processPSVTasksKernel(
    const SA_t* __restrict__ sa_array,
    SA_t* __restrict__ results,
    const size_t* __restrict__ positions,
    const size_t num_tasks,
    const size_t start_pos,
    const size_t end_pos
);

template<typename SA_t>
__global__ void processNSVTasksKernel(
    const SA_t* __restrict__ sa_array,
    SA_t* __restrict__ results,
    const size_t* __restrict__ positions,
    const size_t num_tasks,
    const size_t start_pos,
    const size_t end_pos
);

template<typename SA_t>
__global__ void computePSVNSVKernel(
    const SA_t* __restrict__ input,
    SA_t* __restrict__ psv_output,
    SA_t* __restrict__ nsv_output,
    SA_t* __restrict__ block_min_output,
    const size_t length
);

template<typename SA_t>
__global__ void computePSVNSVKernelTextOrder(
    const SA_t* __restrict__ input,
    SA_t* __restrict__ psv_output,
    SA_t* __restrict__ nsv_output,
    const size_t length
);

template<typename SA_t>
__global__ void processPSVNSVBoundariesKernel(
    const SA_t* __restrict__ sa_array,
    SA_t* __restrict__ psv_sa_order,
    SA_t* __restrict__ nsv_sa_order,
    const size_t length,
    const size_t block_size
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

class PipelinePSVNSVProcessor {
public:
    PipelinePSVNSVProcessor();
    
    template<typename SA_t>
    void process(const SA_t* sa_array, const uint8_t* data, size_t length, const std::string& output_prefix);

     template<typename SA_t>
    void processFullGPUWithGPUSA(SA_t* d_sa_array, const uint8_t* data, size_t length, const std::string& output_prefix);

private:
    GPUProfiler profiler;
    size_t available_memory;

    void calculateAvailableMemory();
    bool canProcessFullGPU(size_t length);

    template<typename SA_t>
    void processFullGPU(const SA_t* sa_array, const uint8_t* data, size_t length, const std::string& output_prefix);

    template<typename SA_t>
    void processWithStreams(const SA_t* sa_array, const uint8_t* data, size_t length, const std::string& output_prefix);

    template<typename SA_t>
    void processTextOrder(const SA_t* sa_array, const std::vector<SA_t>& psv_results,
                        const std::vector<SA_t>& nsv_results, const std::string& output_prefix,
                        size_t length, const uint8_t* data);

    template<typename SA_t>
    void rearrangeTextOrder(const SA_t* sa_array, SA_t* psv, SA_t* nsv,
                            const std::string& output_prefix, size_t length, const uint8_t* data);
    
    template<typename SA_t>
    void convertPSVNSVToTextOrderGPU(SA_t* d_sa_array, SA_t* d_psv_array, SA_t* d_nsv_array, size_t length);

    template<typename SA_t>
    std::pair<std::pair<size_t, size_t>, size_t> LZFactor(const uint8_t *data, size_t i, SA_t psv, SA_t nsv, size_t n);

    template<typename SA_t>
    void ComputeLZ77(const uint8_t *data, SA_t *d_psv_text, SA_t *d_nsv_text, size_t n, std::string file_name);

};

#endif // LZ77_PROCESSOR_CUH