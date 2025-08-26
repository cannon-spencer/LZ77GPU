#ifndef LZ77_PROCESSOR_CUH
#define LZ77_PROCESSOR_CUH

#include <vector>
#include <string>
#include <cstdint>
#include <cuda_runtime.h>

constexpr size_t DEFAULT_BLOCK_SIZE = 256;
constexpr float MEMORY_RESERVE_RATIO = 0.9f;


// CUDA kernel
__global__ void processPSVTasksKernel(
    const size_t* __restrict__ sa_array,
    size_t* __restrict__ results,
    const size_t* __restrict__ positions,
    const size_t num_tasks,
    const size_t start_pos,
    const size_t end_pos
);

__global__ void processNSVTasksKernel(
    const size_t* __restrict__ sa_array,
    size_t* __restrict__ results,
    const size_t* __restrict__ positions,
    const size_t num_tasks,
    const size_t start_pos,
    const size_t end_pos
);

__global__ void computePSVNSVKernel(
    const size_t* __restrict__ input,
    size_t* __restrict__ psv_output,
    size_t* __restrict__ nsv_output,
    const size_t length
);

__global__ void computePSVKernel(
    const size_t* __restrict__ input,
    size_t* __restrict__ psv_output,
    const size_t length
);

__global__ void computeNSVKernel(
    const size_t* __restrict__ input,
    size_t* __restrict__ nsv_output,
    const size_t length
);

__global__ void processPSVBoundariesKernel(
    size_t* __restrict__ psv_output,
    const size_t* __restrict__ input,
    const size_t length,
    const size_t block_size
);

__global__ void processNSVBoundariesKernel(
    size_t* __restrict__ nsv_output,
    const size_t* __restrict__ input,
    const size_t length,
    const size_t block_size
);

__global__ void textOrderMapping(
    const size_t* sa_array,
    const size_t* input,
    size_t* output,
    size_t length
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
    void process(const size_t* sa_array, uint8_t* data, size_t length, const std::string& output_prefix);

private:
    GPUProfiler profiler;
    size_t available_memory;

    void calculateAvailableMemory();
    bool canProcessFullGPU(size_t length);
    void processFullGPU(const size_t* sa_array, uint8_t* data, size_t length, const std::string& output_prefix);
    void processWithStreams(const size_t* sa_array, uint8_t* data, size_t length, const std::string& output_prefix);
    void processTextOrder(const size_t* sa_array, const std::vector<size_t>& psv_results, 
                         const std::vector<size_t>& nsv_results, const std::string& output_prefix, 
                         size_t length, uint8_t* data);
    void rearrangeTextOrder(const size_t* sa_array, size_t* psv, size_t* nsv, 
                           const std::string& output_prefix, size_t length, uint8_t* data);
    std::pair<std::pair<size_t, size_t>, size_t> LZFactor(uint8_t *data, size_t i, size_t psv, size_t nsv, size_t n);
    void ComputeLZ77(uint8_t *data, size_t *d_psv_text, size_t *d_nsv_text, size_t n, std::string file_name);
};

#endif // LZ77_PROCESSOR_CUH