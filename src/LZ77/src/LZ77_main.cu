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

#include "libcubwt.cuh"


// default parameters
constexpr size_t WARP_SIZE = 32;
constexpr size_t DEFAULT_BLOCK_SIZE = 256;
constexpr float MEMORY_RESERVE_RATIO = 0.9f;

class GPUProfiler {
public:
    GPUProfiler() {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
    }

    ~GPUProfiler() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    void start() {
        cudaEventRecord(start_event);
    }

    float stop(const char* operation_name) {
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start_event, stop_event);
        std::cout << operation_name << " took " << milliseconds << " ms\n";
        return milliseconds;
    }

private:
    cudaEvent_t start_event, stop_event;
};

//Used for merging in large file(Stream mode)
__global__ void processPSVTasksKernel(
    const size_t* __restrict__ sa_array,
    size_t* __restrict__ results,
    const size_t* __restrict__ positions,
    const size_t num_tasks,
    const size_t start_pos,
    const size_t end_pos
) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_tasks) return;

    const size_t pos = positions[tid];
    const size_t current = sa_array[pos];
    
    for (size_t j = end_pos; j-- > start_pos;) {
        if (sa_array[j] < current) {
            results[tid] = sa_array[j];
            return;
        }
    }
    results[tid] = SIZE_MAX;
}

__global__ void processNSVTasksKernel(
    const size_t* __restrict__ sa_array,
    size_t* __restrict__ results,
    const size_t* __restrict__ positions,
    const size_t num_tasks,
    const size_t start_pos,
    const size_t end_pos
) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_tasks) return;

    const size_t pos = positions[tid];
    const size_t current = sa_array[pos];
    
    for (size_t j = start_pos; j < end_pos; ++j) {
        if (sa_array[j] < current) {
            results[tid] = sa_array[j];
            return;
        }
    }
    results[tid] = SIZE_MAX;
}

//Used in stream processing
__global__ void computePSVNSVKernel(
    const size_t* __restrict__ input,
    size_t* __restrict__ psv_output,
    size_t* __restrict__ nsv_output,
    const size_t length) 
{
    extern __shared__ size_t shared_data[];
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gid = bid * blockDim.x + tid;

    if (gid < length) {
        psv_output[gid] = SIZE_MAX;
        nsv_output[gid] = SIZE_MAX;
    }

    if (gid < length) {
        shared_data[tid] = input[gid];
    } else {
        shared_data[tid] = SIZE_MAX;
    }
    __syncthreads();

    if (gid < length) {
        const size_t current = shared_data[tid];
        
        for(int i = tid - 1; i >= 0; --i) {
            if(shared_data[i] < current) {
                psv_output[gid] = shared_data[i];
                break;
            }
        }
        
        for(int i = tid + 1; i < blockDim.x && i < length; ++i) {
            if(shared_data[i] < current) {
                nsv_output[gid] = shared_data[i];
                break;
            }
        }
    }
}


//Used in full GPU processing
__global__ void computePSVKernel(const size_t* __restrict__ input,
                               size_t* __restrict__ psv_output,
                               const size_t length) {
    extern __shared__ size_t shared_data[];

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gid = bid * blockDim.x + tid;

    //initialized
    if (gid < length) {
        psv_output[gid] = SIZE_MAX;
    }

    if (gid < length) {
        shared_data[tid] = input[gid];
    } else {
        shared_data[tid] = SIZE_MAX;
    }
    __syncthreads();

    if (gid < length) {
        size_t current = shared_data[tid];

        for (int i = tid - 1; i >= 0; --i) {
            if (shared_data[i] < current) {
                psv_output[gid] = shared_data[i];
                break;
            }
        }
    }
}

//Used in full GPU processing
__global__ void computeNSVKernel(const size_t* __restrict__ input,
                               size_t* __restrict__ nsv_output,
                               const size_t length) {
    extern __shared__ size_t shared_data[];

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gid = bid * blockDim.x + tid;

    if (gid < length) {
        nsv_output[gid] = SIZE_MAX;
    }

    if (gid < length) {
        shared_data[tid] = input[gid];
    } else {
        shared_data[tid] = SIZE_MAX;
    }
    __syncthreads();

    if (gid < length) {
        size_t current = shared_data[tid];

        for (int i = tid + 1; i < blockDim.x && i < length; ++i) {
            if (shared_data[i] < current) {
                nsv_output[gid] = shared_data[i];
                break;
            }
        }
    }
}

//used in full GPU processing
__global__ void processPSVBoundariesKernel(
        size_t* __restrict__ psv_output,
        const size_t* __restrict__ input,
        const size_t length,
        const size_t block_size) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < length) {
        size_t current = input[gid];
        const int current_block = gid / block_size;

        if (current_block > 0 && psv_output[gid] == SIZE_MAX) {
            size_t block_start = (gid / block_size) * block_size;
            for (size_t i = block_start - 1; i != (size_t)-1; --i) {
                if (input[i] < current) {
                    psv_output[gid] = input[i];
                    break;
                }
            }
        }
    }
}

// used in fullGPU
__global__ void processNSVBoundariesKernel(
        size_t* __restrict__ nsv_output,
        const size_t* __restrict__ input,
        const size_t length,
        const size_t block_size) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < length) {
        size_t current = input[gid];
        const int total_blocks = (length + block_size - 1) / block_size;
        const int current_block = gid / block_size;

        if (current_block < total_blocks - 1 && nsv_output[gid] == SIZE_MAX) {
            size_t block_end = ((gid / block_size) + 1) * block_size;
            for (size_t i = block_end; i < length; ++i) {
                if (input[i] < current) {
                    nsv_output[gid] = input[i];
                    break;
                }
            }
        }
    }
}

__global__ void textOrderMapping(
    const size_t* sa_array,
    const size_t* input,
    size_t* output,
    size_t length
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        size_t pos = sa_array[idx];
        output[pos] = input[idx];
    }
}


class PipelinePSVNSVProcessor {
private:
    GPUProfiler profiler;
    size_t available_memory;

    void calculateAvailableMemory() {
        size_t free_memory, total_memory;
        cudaMemGetInfo(&free_memory, &total_memory);
        available_memory = static_cast<size_t>(free_memory * MEMORY_RESERVE_RATIO);
    }

    bool canProcessFullGPU(size_t length) {
        size_t full_gpu_memory = length * sizeof(size_t) * 2;  // psv and nsv
        size_t text_order_memory = length * sizeof(size_t) * 3;

        size_t peak_memory = std::max(full_gpu_memory, text_order_memory);
        peak_memory += DEFAULT_BLOCK_SIZE * sizeof(size_t);

        return peak_memory <= available_memory;
    }

    // CODE FROM KKP3
    size_t LZFactor(uint8_t *data, size_t i, size_t psv, size_t nsv, size_t n, size_t &pos, size_t &len) {
        len = 0;

        auto matchLength = [&](size_t baseIdx) {
            size_t l = 0;
            while (i + l < n && baseIdx + l < n && data[baseIdx + l] == data[i + l]) {
                ++l;
            }
            return l;
        };

        if(nsv == SIZE_MAX) {
            len = matchLength(psv);
            pos = psv;
        }
        else if(psv == SIZE_MAX) {
            len = matchLength(nsv);
            pos = nsv;
        }
        else {
            size_t commonLen = 0;
            while (i + commonLen < n && psv + commonLen < n && nsv + commonLen < n &&
                   data[psv + commonLen] == data[nsv + commonLen]) {
                ++commonLen;
                   }
            len = commonLen;

            if (i + len < n && data[i + len] == data[psv + len]) {
                ++len;
                len += matchLength(psv + len);
                pos = psv;
            } else {
                len += matchLength(nsv + len);
                pos = nsv;
            }
        }

        if (len == 0) pos = data[i];

        return i + std::max((size_t)1, len);
    }

    void ComputeLZ77(uint8_t *data, size_t *d_psv_text, size_t *d_nsv_text, size_t n, std::string file_name) {
        size_t i = 0;
        std::vector<std::pair<size_t, size_t>> buffer;

        while(i < n) {
            size_t pos, len;
            i = LZFactor(data, i, d_psv_text[i + 1] - 1, d_nsv_text[i + 1] - 1, n, pos, len);
            buffer.push_back(std::make_pair(pos, len));
        //    std::cout << "pos: " << pos << ", len: " << len << std::endl;
        }
        printf("LZ77 compression successful\n");

        std::ofstream out_file(file_name, std::ios::binary);
        for (const auto &lz : buffer) {
            out_file.write(reinterpret_cast<const char*>(&lz.first), sizeof(size_t));
            out_file.write(reinterpret_cast<const char*>(&lz.second), sizeof(size_t));
        }
        out_file.close();
    }

void rearrangeTextOrder(const size_t* sa_array,
                       size_t* psv, 
                       size_t* nsv, 
                       const std::string& output_prefix,
                       size_t length,
                       uint8_t* data) 
{
    #pragma omp parallel
    {
        #pragma omp for
        for(size_t i = 0; i < length; i++) {
            if(psv[i] != SIZE_MAX) {
                psv[i] |= (1ULL << 63);
            }
            if(nsv[i] != SIZE_MAX) {
                nsv[i] |= (1ULL << 63);
            }
        }

        #pragma omp for schedule(dynamic)
        for(size_t i = 0; i < length; i++) {
            // 处理PSV
            if(psv[i] != SIZE_MAX && (psv[i] & (1ULL << 63))) {
                size_t curr_pos = i;
                size_t curr_val = psv[i] & ~(1ULL << 63);  // 获取实际值
                
                while(true) {
                    size_t next_pos = sa_array[curr_pos];
                    if(!(psv[next_pos] & (1ULL << 63))) {  // 如果下一个位置已处理
                        break;
                    }
                    
                    size_t next_val = psv[next_pos] & ~(1ULL << 63);
                    psv[next_pos] = (curr_val == (SIZE_MAX & ~(1ULL << 63))) ? SIZE_MAX : curr_val;
                    
                    if(next_pos == i) break;
                    
                    curr_pos = next_pos;
                    curr_val = next_val;
                }
            }

            // 处理NSV
            if(nsv[i] != SIZE_MAX && (nsv[i] & (1ULL << 63))) {
                size_t curr_pos = i;
                size_t curr_val = nsv[i] & ~(1ULL << 63);
                
                while(true) {
                    size_t next_pos = sa_array[curr_pos];
                    if(!(nsv[next_pos] & (1ULL << 63))) {
                        break;
                    }
                    
                    size_t next_val = nsv[next_pos] & ~(1ULL << 63);
                     nsv[next_pos] = (curr_val == (SIZE_MAX & ~(1ULL << 63))) ? SIZE_MAX : curr_val;
                    
                    if(next_pos == i) break;
                    
                    curr_pos = next_pos;
                    curr_val = next_val;
                }
            }
        }

        #pragma omp for schedule(static)
        for(size_t i = 0; i < length; i++) {
            if(psv[i] != SIZE_MAX && (psv[i] & (1ULL << 63))) {
                psv[i] &= ~(1ULL << 63);
            }
            if(nsv[i] != SIZE_MAX && (nsv[i] & (1ULL << 63))) {
                nsv[i] &= ~(1ULL << 63);
            }
        }
    }

    // std::cout << "PSV after rearrangement: ";
    // for(size_t i = 0; i < length; i++) {
    //     std::cout << psv[i] << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "NSV after rearrangement: ";
    // for(size_t i = 0; i < length; i++) {
    //     std::cout << nsv[i] << " ";
    // }
    // std::cout << std::endl;

    profiler.start();
    std::string lz_output = output_prefix + "_lz77.bin";
    ComputeLZ77(data, psv, nsv, length - 1, lz_output);
    profiler.stop("LZ77 Processing");
}

public:
    PipelinePSVNSVProcessor()
    {
        calculateAvailableMemory();
        std::cout << "After memory calculated" << std::endl;
    }

 void processTextOrder(const size_t* sa_array,
                         const std::vector<size_t>& psv_results,
                         const std::vector<size_t>& nsv_results,
                         const std::string& output_prefix,
                         size_t length,
                         uint8_t* data) {
        profiler.start();

        std::vector<size_t> psv_text_order(length, SIZE_MAX);
        std::vector<size_t> nsv_text_order(length, SIZE_MAX);

        const size_t BLOCK_SIZE = DEFAULT_BLOCK_SIZE;

        if (canProcessFullGPU(length)) {
            size_t *d_sa_array, *d_output, *d_input;
            try{
                cudaMalloc(&d_sa_array, length * sizeof(size_t));
                cudaMalloc(&d_output, length * sizeof(size_t));
                cudaMalloc(&d_input, length * sizeof(size_t));
                cudaMemcpy(d_sa_array, sa_array, length * sizeof(size_t), cudaMemcpyHostToDevice);

            {
                    cudaMemcpy(d_input, psv_results.data(), length * sizeof(size_t), cudaMemcpyHostToDevice);

                    textOrderMapping<<<(length + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                        d_sa_array, d_input, d_output, length
                    );

                    cudaMemcpy(psv_text_order.data(), d_output, length * sizeof(size_t), cudaMemcpyDeviceToHost);
                    
            }

            {
                cudaMemcpy(d_input, nsv_results.data(), length * sizeof(size_t), cudaMemcpyHostToDevice);
                
                textOrderMapping<<<(length + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE>>>(
                    d_sa_array, d_input, d_output, length
                );
                
                cudaMemcpy(nsv_text_order.data(), d_output, length * sizeof(size_t), cudaMemcpyDeviceToHost);
            }

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string("CUDA error in text order processing: ") + 
                                      cudaGetErrorString(err));
            }

            }
            catch(const std::exception& e)
            {
                std::cerr << e.what() << '\n';
            }

            cudaFree(d_sa_array);
            cudaFree(d_output);
            cudaFree(d_input);
            

        } else {
            #pragma omp parallel for
            for(size_t i = 0; i < length; i++) {
                psv_text_order[sa_array[i]] = psv_results[i];
                nsv_text_order[sa_array[i]] = nsv_results[i];
            }
        }

        profiler.stop("Text Order Processing");

        profiler.start();
        std::string lz_output = output_prefix + "_lz77.bin";
        ComputeLZ77(data, psv_text_order.data(), nsv_text_order.data(), length - 1, lz_output);
        profiler.stop("LZ77 Processing");
    }

    void processFullGPU(const size_t* sa_array, uint8_t* data, size_t length, const std::string& output_prefix) {
        profiler.start();

        std::vector<size_t> h_psv_results(length);
        std::vector<size_t> h_nsv_results(length);

        const int block_size = DEFAULT_BLOCK_SIZE;
        const int num_blocks = (length + block_size - 1) / block_size;
        const size_t shared_mem_size = block_size * sizeof(size_t);

        //allocate device memory
        size_t *d_input, *d_output;
        cudaMalloc(&d_input, length * sizeof(size_t));
        cudaMalloc(&d_output, length * sizeof(size_t));

        // copy input data to device
        cudaMemcpy(d_input, sa_array, length * sizeof(size_t), cudaMemcpyHostToDevice);

        // 
        {
        computePSVKernel<<<num_blocks, block_size, shared_mem_size>>>(
            d_input, d_output, length
        );

        processPSVBoundariesKernel<<<num_blocks, block_size>>>(
            d_output, d_input, length, block_size
        );

        cudaMemcpy(h_psv_results.data(), d_output, length * sizeof(size_t), cudaMemcpyDeviceToHost);
        }

        // 
        {
        computeNSVKernel<<<num_blocks, block_size, shared_mem_size>>>(
            d_input, d_output, length
        );

        processNSVBoundariesKernel<<<num_blocks, block_size>>>(
            d_output, d_input, length, block_size
        );

        cudaMemcpy(h_nsv_results.data(), d_output, length * sizeof(size_t), cudaMemcpyDeviceToHost);
        }

        cudaFree(d_input);
        cudaFree(d_output);

        profiler.stop("Full GPU Processing");

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error("CUDA error occurred during GPU processing");
        }
        processTextOrder(sa_array, h_psv_results, h_nsv_results, output_prefix, length, data);
    }

    void process(const size_t* sa_array, uint8_t* data, size_t length, const std::string& output_prefix) {
        try {
            std::cout << "Available GPU memory: " << available_memory << " bytes" << std::flush << std::endl;
            std::cout << "Input length: " << length << " bytes" << std::flush << std::endl;
            //canProcessFullGPU(length)
            if (canProcessFullGPU(length)) {
                std::cout << "Using full GPU processing mode" << std::flush << std::endl;
                processFullGPU(sa_array, data, length, output_prefix);
            } else {
                std::cout << "Using stream processing mode" << std::flush << std::endl;
                processWithStreams(sa_array, data, length, output_prefix);
            }
        } catch (const std::bad_alloc& e) {
            std::cerr << "Memory allocation failed: " << e.what() << std::endl;
            throw;
        }
    }

     void processWithStreams(const size_t* sa_array, uint8_t* data, size_t length, const std::string& output_prefix) {
        try {
            std::cout << "Starting stream processing..." << std::flush << std::endl;
            profiler.start();

            size_t max_batch_size = std::min(
                (available_memory / (3 * sizeof(size_t))),
                static_cast<size_t>(256 * 1024 * 1024)  // 1GB batch size
            );


            max_batch_size = (max_batch_size / DEFAULT_BLOCK_SIZE) * DEFAULT_BLOCK_SIZE;
            size_t optimal_batch_size = std::min(length, max_batch_size);
//TODO      ONLY FOR TESTING    
            // optimal_batch_size = 8; 
            size_t total_blocks = (length + optimal_batch_size - 1) / optimal_batch_size;


            std::cout << "Total blocks: " << total_blocks << std::endl;
            std::cout << "Final batch size: " << optimal_batch_size/(1024*1024) << "MB" << std::flush << std::endl;

            std::vector<size_t> final_psv_results(length);
            std::vector<size_t> final_nsv_results(length);

           { 
            cudaStream_t compute_stream;
            cudaStreamCreate(&compute_stream);

            const int block_size = DEFAULT_BLOCK_SIZE;
            const size_t shared_mem_size = block_size * sizeof(size_t);

            size_t* d_input, *d_psv_output, *d_nsv_output;
            cudaMalloc(&d_input, optimal_batch_size * sizeof(size_t));
            cudaMalloc(&d_psv_output, optimal_batch_size * sizeof(size_t));
            cudaMalloc(&d_nsv_output, optimal_batch_size * sizeof(size_t));

            size_t processed_blocks = 0;
            for (size_t offset = 0; offset < length; offset += optimal_batch_size)
            {

                processed_blocks++;
                float progress = (processed_blocks * 100.0f) / total_blocks;
                std::cout << "\rGPU Processing: " << std::fixed << std::setprecision(2) 
                         << progress << "% [Block " << processed_blocks << "/" << total_blocks << "]" 
                         << std::flush;


                size_t current_batch = std::min(optimal_batch_size, length - offset);
                const int num_blocks = (current_batch + block_size - 1) / block_size;

                cudaMemcpyAsync(d_input, sa_array + offset, 
                            current_batch * sizeof(size_t),
                            cudaMemcpyHostToDevice, 
                            compute_stream);

                computePSVNSVKernel<<<num_blocks, block_size, shared_mem_size, compute_stream>>>(
                    d_input, d_psv_output, d_nsv_output, current_batch
                );

                cudaMemcpyAsync(&final_psv_results[offset], d_psv_output,
                              current_batch * sizeof(size_t),
                              cudaMemcpyDeviceToHost,
                              compute_stream);
                cudaMemcpyAsync(&final_nsv_results[offset], d_nsv_output,
                              current_batch * sizeof(size_t),
                              cudaMemcpyDeviceToHost,
                              compute_stream);

                cudaStreamSynchronize(compute_stream);
            }

            cudaStreamDestroy(compute_stream);
            cudaFree(d_input);
            cudaFree(d_psv_output);
            cudaFree(d_nsv_output);
            }
            

            profiler.stop("GPU Stream Processing");

        profiler.start();

         const size_t num_blocks = (length + optimal_batch_size - 1) / optimal_batch_size;
        
        std::vector<std::vector<size_t>> block_psv_tasks(num_blocks);
        std::vector<std::vector<size_t>> block_nsv_tasks(num_blocks);
        std::atomic<size_t> processed_blocks{0};
        
        #pragma omp parallel for schedule(dynamic)
        for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
            size_t block_start = block_idx * optimal_batch_size;
            size_t block_end = std::min(block_start + optimal_batch_size, length);
            
            if (block_idx > 0) {
                for (size_t i = block_start; i < block_end; ++i) {
                    if (final_psv_results[i] == SIZE_MAX) {
                        block_psv_tasks[block_idx].push_back(i);
                    }
                }
            }

            if (block_idx < num_blocks - 1) {
                for (size_t i = block_start; i < block_end; ++i) {
                    if (final_nsv_results[i] == SIZE_MAX) {
                        block_nsv_tasks[block_idx].push_back(i);
                    }
                }
            }
        }

        size_t total_tasks = 0;
        for (const auto& tasks : block_psv_tasks) total_tasks += tasks.size();
        for (const auto& tasks : block_nsv_tasks) total_tasks += tasks.size();
        

        std::cout << "Total merge tasks: " << total_tasks << std::endl;
        
        if (total_tasks > 0)
        {
            cudaStream_t task_stream;
            cudaStreamCreate(&task_stream);

            size_t *d_sa_array, *d_positions, *d_results;

            try
            {
                cudaMalloc(&d_sa_array, length * sizeof(size_t));
                cudaMemcpyAsync(d_sa_array, sa_array, length * sizeof(size_t),
                       cudaMemcpyHostToDevice, task_stream);

                const size_t MAX_BATCH_TASKS = 1024 * 1024;
                // const size_t MAX_BATCH_TASKS = 4;
                std::atomic<size_t> total_processed_tasks{0};

                for (size_t block_idx = 0; block_idx < num_blocks; block_idx++)
                {
                    std::vector<size_t>& tasks = block_psv_tasks[block_idx];
                    if (tasks.empty()) continue;

                    std::cout << "\nProcessing PSV block " << block_idx + 1 << "/" << num_blocks 
                              << " with " << tasks.size() << " tasks" << std::endl;

                    for (size_t task_offset = 0; task_offset < tasks.size(); task_offset += MAX_BATCH_TASKS) 
                    {   
                        size_t current_batch_size = std::min(MAX_BATCH_TASKS, tasks.size() - task_offset);
                        float batch_progress = (task_offset + current_batch_size) * 100.0f / tasks.size();
                        std::cout << "\rBlock progress: " << std::fixed << std::setprecision(2) 
                                  << batch_progress << "% [Batch: " << task_offset/MAX_BATCH_TASKS + 1 
                                  << "/" << (tasks.size() + MAX_BATCH_TASKS - 1)/MAX_BATCH_TASKS << "]" 
                                  << std::flush;

                        
                        cudaMalloc(&d_positions, current_batch_size * sizeof(size_t));
                        cudaMalloc(&d_results, current_batch_size * sizeof(size_t));

                        cudaMemcpyAsync(d_positions, tasks.data() + task_offset,
                                    current_batch_size * sizeof(size_t),
                                    cudaMemcpyHostToDevice, task_stream);

                        const size_t BLOCK_SIZE = DEFAULT_BLOCK_SIZE;
                        size_t num_gpu_blocks = (current_batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
                        
                        size_t start_pos = 0;
                        size_t end_pos = block_idx * optimal_batch_size;

                        processPSVTasksKernel<<<num_gpu_blocks, BLOCK_SIZE, 0, task_stream>>>(
                            d_sa_array, d_results, d_positions,
                            current_batch_size, start_pos, end_pos
                        );

                        std::vector<size_t> task_results(current_batch_size);
                        cudaMemcpyAsync(task_results.data(), d_results,
                                    current_batch_size * sizeof(size_t),
                                    cudaMemcpyDeviceToHost, task_stream);
                        cudaStreamSynchronize(task_stream);

                        for (size_t i = 0; i < current_batch_size; ++i) {
                            if (task_results[i] != SIZE_MAX) {
                                final_psv_results[tasks[task_offset + i]] = task_results[i];
                            }
                        }

                        cudaFree(d_positions);
                        cudaFree(d_results);
                        d_positions = nullptr;
                        d_results = nullptr;

                        total_processed_tasks += current_batch_size;
                        float total_progress = total_processed_tasks * 100.0f / total_tasks;
                        std::cout << " | Total: " << std::fixed << std::setprecision(2) 
                                  << total_progress << "% [" << total_processed_tasks << "/" << total_tasks << "]" 
                                  << std::flush;
                    }
                    std::cout << std::endl;
                }

                std::cout << "\nPSV merge completed" << std::endl;

                for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
                    std::vector<size_t>& tasks = block_nsv_tasks[block_idx];
                    if (tasks.empty()) continue;

                    std::cout << "\nProcessing NSV block " << block_idx + 1 << "/" << num_blocks 
                              << " with " << tasks.size() << " tasks" << std::endl;
                    for (size_t task_offset = 0; task_offset < tasks.size(); task_offset += MAX_BATCH_TASKS) 
                    {
                        size_t current_batch_size = std::min(MAX_BATCH_TASKS, tasks.size() - task_offset);
        
                        float batch_progress = (task_offset + current_batch_size) * 100.0f / tasks.size();
                        std::cout << "\rBlock progress: " << std::fixed << std::setprecision(2) 
                                  << batch_progress << "% [Batch: " << task_offset/MAX_BATCH_TASKS + 1 
                                  << "/" << (tasks.size() + MAX_BATCH_TASKS - 1)/MAX_BATCH_TASKS << "]" 
                                  << std::flush;
                        
                        cudaMalloc(&d_positions, current_batch_size * sizeof(size_t));
                        cudaMalloc(&d_results, current_batch_size * sizeof(size_t));

                        cudaMemcpyAsync(d_positions, tasks.data() + task_offset,
                                    current_batch_size * sizeof(size_t),
                                    cudaMemcpyHostToDevice, task_stream);

                        const size_t BLOCK_SIZE = DEFAULT_BLOCK_SIZE;
                        size_t num_gpu_blocks = (current_batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
                        
                        size_t start_pos = (block_idx + 1) * optimal_batch_size;
                        size_t end_pos = length;

                        processNSVTasksKernel<<<num_gpu_blocks, BLOCK_SIZE, 0, task_stream>>>(
                            d_sa_array, d_results, d_positions,
                            current_batch_size, start_pos, end_pos
                        );

                        std::vector<size_t> task_results(current_batch_size);
                        cudaMemcpyAsync(task_results.data(), d_results,
                                    current_batch_size * sizeof(size_t),
                                    cudaMemcpyDeviceToHost, task_stream);
                        cudaStreamSynchronize(task_stream);

                        for (size_t i = 0; i < current_batch_size; ++i) {
                            if (task_results[i] != SIZE_MAX) {
                                final_nsv_results[tasks[task_offset + i]] = task_results[i];
                            }
                        }

                        cudaFree(d_positions);
                        cudaFree(d_results);
                        d_positions = nullptr;
                        d_results = nullptr;

                        total_processed_tasks += current_batch_size;
                        float total_progress = total_processed_tasks * 100.0f / total_tasks;
                        std::cout << " | Total: " << std::fixed << std::setprecision(2) 
                                  << total_progress << "% [" << total_processed_tasks << "/" << total_tasks << "]" 
                                  << std::flush;
                    }
                    std::cout << std::endl;
                }

                std::cout << "\nNSV merge completed" << std::endl;
                
            }
            catch(const std::exception& e)
            {
                if (d_sa_array) cudaFree(d_sa_array);
                if (d_positions) cudaFree(d_positions);
                if (d_results) cudaFree(d_results);
                cudaStreamDestroy(task_stream);
                throw;
            }
             
            cudaFree(d_sa_array);
            cudaStreamDestroy(task_stream);
        }

        profiler.stop("GPU Merge");  

        // print the final psv and nsv results
        // std::cout << "PSV Results:" << std::endl;
        // for (size_t i = 0; i < final_psv_results.size(); ++i) {
        //     std::cout << final_psv_results[i] << " ";
        // }
        // std::cout << std::endl;

        // std::cout << "NSV Results:" << std::endl;
        // for (size_t i = 0; i < final_nsv_results.size(); ++i) {
        //     std::cout << final_nsv_results[i] << " ";
        // }
        // std::cout << std::endl;
        

        // processTextOrder(sa_array, final_psv_results, final_nsv_results, output_prefix, length, data);
        rearrangeTextOrder(sa_array, final_psv_results.data(), final_nsv_results.data(), output_prefix, length, data);
        } catch (const std::exception& e) {
            std::cerr << "Error in stream processing: " << e.what() << std::endl;
            throw;
        }
    }

};

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
    std::vector <uint8_t> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    if (data.empty()) {
        std::cerr << "File is empty or could not be read correctly." << std::endl;
        return 1;
    }


    /* Create Suffix Array and Inverse Suffix Array using lib_cubwt */
    data.push_back(0);
    size_t length = data.size();
    size_t* SA = new size_t[length];

    GPUProfiler profiler;
    profiler.start();

    if(length <= UINT32_MAX){
        // allocate device storage
        void *device_storage = nullptr;
        int result = libcubwt_allocate_device_storage(&device_storage, length);

        if (result == LIBCUBWT_NO_ERROR){
            // generate suffix array
            uint32_t* temp_SA = new uint32_t[length];
            result = libcubwt_sa(device_storage, data.data(), temp_SA, length);
            if (result != LIBCUBWT_NO_ERROR) {
                std::cerr << "Failed to generate suffix array: " << result << std::endl;
                libcubwt_free_device_storage(device_storage);
                delete[] SA;
                return 1;
            }
            libcubwt_free_device_storage(device_storage);
            // Copy to SA (ensure types match)
            for (size_t i = 0; i < length; ++i) {
                SA[i] = static_cast<size_t>(temp_SA[i]);
            }

            delete[] temp_SA;
            std::cout << "Used libcubwt for SA construction" << std::endl;

        }
        else{
            std::cout << "GPU memory allocation failed (error: " << result << "), switching to SDSL" << std::endl;
            try {
                {
                    sdsl::int_vector<sizeof(size_t) * 8> sdsl_sa(length);
                    std::cout << "Input file size: " << length << " bytes" << std::endl;

                    sdsl::algorithm::calculate_sa(static_cast<const unsigned char *>(data.data()), length, sdsl_sa);
                    std::memcpy(SA, sdsl_sa.data(), length * sizeof(size_t));
                }

//used for local debug
//                {
//                    sdsl::int_vector<64> sdsl_sa(length);
//                    std::cout << "Input file size: " << length << " bytes" << std::endl;
//
//                    sdsl::algorithm::calculate_sa(static_cast<const unsigned char*>(data.data()), length, sdsl_sa);
//                    std::cout<<"SA computed"<<std::endl;
//                    for (size_t i = 0; i < length; ++i) {
//                        SA[i] = sdsl_sa[i];
//                    }
//                }

                std::cout << "SA construction finished" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Failed to construct suffix array using SDSL: " << e.what() << std::endl;
                delete[] SA;
                return 1;
            }
        }
    }else{
        try {
            {
                sdsl::int_vector<sizeof(size_t) * 8> sdsl_sa(length);

                std::cout << "Used SDSL for SA construction" << std::endl;
                sdsl::algorithm::calculate_sa(static_cast<const unsigned char *>(data.data()), length, sdsl_sa);
                std::memcpy(SA, sdsl_sa.data(), length * sizeof(size_t));
            }

        } catch (const std::exception& e) {
            std::cerr << "Failed to construct suffix array using SDSL: " << e.what() << std::endl;
            delete[] SA;
            return 1;
        }
    }

    try {
        profiler.stop("Suffix Array Generation");
        std::cout << "Before processor initialized" << std::endl;
        PipelinePSVNSVProcessor processor;
        std::cout << "After processor initialized" << std::endl;
        processor.process(SA, data.data(), length, output_prefix);
    } catch (const std::exception& e) {
        std::cerr << "Error occurred: " << e.what() << std::endl;
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        }
        return 1;
    }
    // Cleanup
    delete[] SA;

    return 0;
}


