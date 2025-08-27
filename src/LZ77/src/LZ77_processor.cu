#include "LZ77_processor.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <atomic>

GPUProfiler::GPUProfiler() : start_event(nullptr), stop_event(nullptr) {
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
}

GPUProfiler::~GPUProfiler() {
    if (start_event) {
        cudaEventDestroy(start_event);
        start_event = nullptr;
    }
    if (stop_event) {
        cudaEventDestroy(stop_event);
        stop_event = nullptr;
    }
}

void GPUProfiler::start() {
    cudaEventRecord(start_event);
}

float GPUProfiler::stop(const char* operation_name) {
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);
    std::cout << operation_name << " took " << milliseconds << " ms\n";
    return milliseconds;
}

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


void PipelinePSVNSVProcessor::calculateAvailableMemory() {
    size_t free_memory, total_memory;
    cudaMemGetInfo(&free_memory, &total_memory);
    available_memory = static_cast<size_t>(free_memory * MEMORY_RESERVE_RATIO);
}

bool PipelinePSVNSVProcessor::canProcessFullGPU(size_t length) {
    size_t full_gpu_memory = length * sizeof(size_t) * 2;  // psv and nsv
    size_t text_order_memory = length * sizeof(size_t) * 3;

    size_t peak_memory = std::max(full_gpu_memory, text_order_memory);
    peak_memory += DEFAULT_BLOCK_SIZE * sizeof(size_t);

    return peak_memory <= available_memory;
}

    // LZ77 Factor computation using PSV and NSV
std::pair<std::pair<size_t, size_t>, size_t> PipelinePSVNSVProcessor::LZFactor(
    uint8_t *data, size_t i, size_t psv, size_t nsv, size_t n) {
    
    size_t len = 0;
    size_t pos = 0;

    // Helper function to compute match length from a given position
    auto matchLength = [&](size_t baseIdx) -> size_t {
        size_t l = 0;
        while (i + l < n && baseIdx + l < n && data[baseIdx + l] == data[i + l]) {
            ++l;
        }
        return l;
    };

    // If both PSV and NSV are invalid, output literal character
    if (psv == SIZE_MAX && nsv == SIZE_MAX) {
        return std::make_pair(std::make_pair(data[i], 0), i + 1);
    }
    
    // Only PSV is valid
    if (nsv == SIZE_MAX) {
        len = matchLength(psv);
        pos = psv;
    }
    // Only NSV is valid  
    else if (psv == SIZE_MAX) {
        len = matchLength(nsv);
        pos = nsv;
    }
    // Both PSV and NSV are valid - choose the one that gives longer match
    else {
        size_t psv_len = matchLength(psv);
        size_t nsv_len = matchLength(nsv);
        
        if (psv_len >= nsv_len) {
            len = psv_len;
            pos = psv;
        } else {
            len = nsv_len;
            pos = nsv;
        }
    }

    // If no match found, output literal character
    if (len == 0) {
        return std::make_pair(std::make_pair(data[i], 0), i + 1);
    }

    // Ensure we advance by at least 1 position
    size_t next_i = i + std::max((size_t)1, len);
    return std::make_pair(std::make_pair(pos, len), next_i);
}

void PipelinePSVNSVProcessor::ComputeLZ77(uint8_t *data, size_t *d_psv_text, size_t *d_nsv_text, size_t n, std::string file_name) {
    size_t i = 0;
    std::vector<std::pair<size_t, size_t>> buffer;
    

    while(i < n) {
        // Use the PSV and NSV values directly for the current position i
        size_t psv = d_psv_text[i];
        size_t nsv = d_nsv_text[i];
        
        auto result = LZFactor(data, i, psv, nsv, n);
        size_t pos = result.first.first;
        size_t len = result.first.second;
        i = result.second;
        
        // printf("LZ Factor at pos %zu - Ref: %zu, Len: %zu\n", i - std::max((size_t)1, len), pos, len);
        buffer.push_back(std::make_pair(pos, len));
    }
    printf("LZ77 compression successful, generated %zu factors\n", buffer.size());

    std::ofstream out_file(file_name, std::ios::binary);
    for (const auto &lz : buffer) {
        out_file.write(reinterpret_cast<const char*>(&lz.first), sizeof(size_t));
        out_file.write(reinterpret_cast<const char*>(&lz.second), sizeof(size_t));
    }
    out_file.close();
}

void PipelinePSVNSVProcessor::rearrangeTextOrder(const size_t* sa_array,
                       size_t* psv, 
                       size_t* nsv, 
                       const std::string& output_prefix,
                       size_t length,
                       uint8_t* data) {
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

            if(psv[i] != SIZE_MAX && (psv[i] & (1ULL << 63))) {
                size_t curr_pos = i;
                size_t curr_val = psv[i] & ~(1ULL << 63);  // 
                
                while(true) {
                    size_t next_pos = sa_array[curr_pos];
                    if(!(psv[next_pos] & (1ULL << 63))) {  
                        break;
                    }
                    
                    size_t next_val = psv[next_pos] & ~(1ULL << 63);
                    psv[next_pos] = (curr_val == (SIZE_MAX & ~(1ULL << 63))) ? SIZE_MAX : curr_val;
                    
                    if(next_pos == i) break;
                    
                    curr_pos = next_pos;
                    curr_val = next_val;
                }
            }

            // Handle the NSV 
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

    profiler.start();
    std::string lz_output = output_prefix + "_lz77.bin";
    ComputeLZ77(data, psv, nsv, length - 1, lz_output);
    profiler.stop("LZ77 Processing");
}

PipelinePSVNSVProcessor::PipelinePSVNSVProcessor() {
    calculateAvailableMemory();
    std::cout << "After memory calculated" << std::endl;
}

    void PipelinePSVNSVProcessor::processFullGPU(const size_t* sa_array, uint8_t* data, size_t length, const std::string& output_prefix) {
        profiler.start();

        std::vector<size_t> h_psv_results(length);
        std::vector<size_t> h_nsv_results(length);

        const int block_size = DEFAULT_BLOCK_SIZE;
        const int num_blocks = (length + block_size - 1) / block_size;
        printf("num_blocks: %d\n", num_blocks);
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
        rearrangeTextOrder(sa_array, h_psv_results.data(), h_nsv_results.data(), output_prefix, length, data);
    }

    void PipelinePSVNSVProcessor::process(const size_t* sa_array, uint8_t* data, size_t length, const std::string& output_prefix) {
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

     void PipelinePSVNSVProcessor::processWithStreams(const size_t* sa_array, uint8_t* data, size_t length, const std::string& output_prefix) {
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

        rearrangeTextOrder(sa_array, final_psv_results.data(), final_nsv_results.data(), output_prefix, length, data);
        } catch (const std::exception& e) {
            std::cerr << "Error in stream processing: " << e.what() << std::endl;
            throw;
        }
    }
