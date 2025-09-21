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

// Template kernel implementations
template<typename SA_t>
__global__ void processPSVTasksKernel(
    const SA_t* __restrict__ sa_array,
    SA_t* __restrict__ results,
    const size_t* __restrict__ positions,
    const size_t num_tasks,
    const size_t start_pos,
    const size_t end_pos
) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_tasks) return;

    const size_t pos = positions[tid];
    const SA_t current = sa_array[pos];
    const SA_t MAX_VAL = get_max_value<SA_t>();
    
    for (size_t j = end_pos; j-- > start_pos;) {
        if (sa_array[j] < current) {
            results[tid] = sa_array[j];
            return;
        }
    }
    results[tid] = MAX_VAL;
}

template<typename SA_t>
__global__ void processNSVTasksKernel(
    const SA_t* __restrict__ sa_array,
    SA_t* __restrict__ results,
    const size_t* __restrict__ positions,
    const size_t num_tasks,
    const size_t start_pos,
    const size_t end_pos
) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_tasks) return;

    const size_t pos = positions[tid];
    const SA_t current = sa_array[pos];
    const SA_t MAX_VAL = get_max_value<SA_t>();
    
    for (size_t j = start_pos; j < end_pos; ++j) {
        if (sa_array[j] < current) {
            results[tid] = sa_array[j];
            return;
        }
    }
    results[tid] = MAX_VAL;
}

template<typename SA_t>
__global__ void computePSVNSVKernel(
    const SA_t* __restrict__ input,
    SA_t* __restrict__ psv_output,
    SA_t* __restrict__ nsv_output,
    SA_t* __restrict__ block_min_output,
    const size_t length) 
{
    extern __shared__ uint8_t shared_mem[];
    SA_t* shared_data = reinterpret_cast<SA_t*>(shared_mem);
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gid = bid * blockDim.x + tid;
    const SA_t MAX_VAL = get_max_value<SA_t>();

    if (gid < length) {
        psv_output[gid] = MAX_VAL;
        nsv_output[gid] = MAX_VAL;
    }

    if (gid < length) {
        shared_data[tid] = input[gid];
    } else {
        shared_data[tid] = MAX_VAL;
    }
    __syncthreads();

    if (gid < length) {
        const SA_t current = shared_data[tid];
        
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

    // Compute block minimum using reduction
    __syncthreads();
    
    // Parallel reduction to find minimum in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] = min(shared_data[tid], shared_data[tid + stride]);
        }
        __syncthreads();
    }
    
    // Thread 0 writes the block minimum
    if (tid == 0) {
        block_min_output[bid] = shared_data[0];
    }
}

template<typename SA_t>
__global__ void processPSVNSVBoundariesKernel(
    const SA_t* __restrict__ sa_array,     
    SA_t* __restrict__ psv_sa_order,
    SA_t* __restrict__ nsv_sa_order,
    const SA_t* __restrict__ block_min_output, 
    const size_t length,
    const size_t block_size) 
{

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gid = bid * blockDim.x + tid;
    const SA_t MAX_VAL = get_max_value<SA_t>();
    const size_t total_blocks = (length + block_size - 1) / block_size;
    
    if (gid >= length) return;

    const SA_t current = sa_array[gid];
    SA_t psv_val = psv_sa_order[gid];
    SA_t nsv_val = nsv_sa_order[gid];

    // Progressive PSV search - start with small chunks, grow exponentially
    if (psv_val == MAX_VAL && bid > 0) {

        int search_block = bid - 1;
        while (search_block >= 0 && block_min_output[search_block] >= current) {
            search_block--;
        }
        

        if (search_block >= 0) {
            size_t search_end = min((size_t)gid, (size_t)(search_block + 1) * block_size);
            size_t search_start = max((size_t)0, search_end > 2048 ? search_end - 2048 : 0);
            
            for (size_t i = search_end - 1; i >= search_start && psv_val == MAX_VAL; --i) {
                if (sa_array[i] < current) {
                    psv_val = sa_array[i];
                    break;
                }
                if (i == 0) break; 
            }
        }
    }
    
    // Progressive NSV search - same strategy
    if (nsv_val == MAX_VAL && bid + 1 < total_blocks) {

        size_t search_block = bid + 1;
        while (search_block < total_blocks && block_min_output[search_block] >= current) {
            search_block++;
        }
        
        if (search_block < total_blocks) {
            size_t search_start = max((size_t)gid + 1, search_block * block_size);
            size_t search_end = min(length, search_start + 2048);
            
            for (size_t i = search_start; i < search_end && nsv_val == MAX_VAL; ++i) {
                if (sa_array[i] < current) {
                    nsv_val = sa_array[i];
                    break;
                }
            }
        }
    }
    psv_sa_order[gid] = psv_val;
    nsv_sa_order[gid] = nsv_val;
}

template<typename SA_t>
void PipelinePSVNSVProcessor::convertToTextOrderWithCUB(
    const SA_t* d_sa_array,
    const SA_t* d_psv_sa_order,
    const SA_t* d_nsv_sa_order,
    SA_t* d_psv_text_order,
    SA_t* d_nsv_text_order,
    size_t length)
{
    auto sa_ptr = thrust::device_pointer_cast(d_sa_array);
    auto psv_sa_ptr = thrust::device_pointer_cast(d_psv_sa_order);
    auto nsv_sa_ptr = thrust::device_pointer_cast(d_nsv_sa_order);
    auto psv_text_ptr = thrust::device_pointer_cast(d_psv_text_order);
    auto nsv_text_ptr = thrust::device_pointer_cast(d_nsv_text_order);

    thrust::scatter(psv_sa_ptr, psv_sa_ptr + length, sa_ptr, psv_text_ptr);
    thrust::scatter(nsv_sa_ptr, nsv_sa_ptr + length, sa_ptr, nsv_text_ptr);

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

template<typename SA_t>
std::pair<std::pair<size_t, size_t>, size_t> PipelinePSVNSVProcessor::LZFactor(
    const uint8_t *data, size_t i, SA_t psv, SA_t nsv, size_t n) {
    
    size_t len = 0;
    size_t pos = 0;
    const SA_t MAX_VAL = get_max_value<SA_t>();

    auto matchLength = [&](SA_t baseIdx) -> size_t {
        if (is_invalid_value(baseIdx)) return 0;
        
        size_t l = 0;
        while (i + l < n && baseIdx + l < n && data[baseIdx + l] == data[i + l]) {
            ++l;
        }
        return l;
    };

    if (psv == MAX_VAL && nsv == MAX_VAL) {
        return std::make_pair(std::make_pair(data[i], 0), i + 1);
    }
    
    if (nsv == MAX_VAL) {
        len = matchLength(psv);
        pos = static_cast<size_t>(psv);
    }
    else if (psv == MAX_VAL) {
        len = matchLength(nsv);
        pos = static_cast<size_t>(nsv);
    }
    else {
        size_t psv_len = matchLength(psv);
        size_t nsv_len = matchLength(nsv);
        
        if (psv_len >= nsv_len) {
            len = psv_len;
            pos = static_cast<size_t>(psv);
        } else {
            len = nsv_len;
            pos = static_cast<size_t>(nsv);
        }
    }

    if (len == 0) {
        return std::make_pair(std::make_pair(data[i], 0), i + 1);
    }

    size_t next_i = i + std::max((size_t)1, len);
    return std::make_pair(std::make_pair(pos, len), next_i);
}

template<typename SA_t>
void PipelinePSVNSVProcessor::ComputeLZ77(const uint8_t *data, SA_t *d_psv_text, SA_t *d_nsv_text, size_t n, std::string file_name) {
    size_t i = 0;
    std::vector<std::pair<size_t, size_t>> buffer;
    
    while(i < n) {
        SA_t psv = d_psv_text[i];
        SA_t nsv = d_nsv_text[i];
        
        auto result = LZFactor(data, i, psv, nsv, n);
        size_t pos = result.first.first;
        size_t len = result.first.second;
        i = result.second;
        
        buffer.push_back(std::make_pair(pos, len));
    }

    std::ofstream out_file(file_name, std::ios::binary);
    for (const auto &lz : buffer) {
        out_file.write(reinterpret_cast<const char*>(&lz.first), sizeof(size_t));
        out_file.write(reinterpret_cast<const char*>(&lz.second), sizeof(size_t));
    }
    out_file.close();
}

template<typename SA_t>
void PipelinePSVNSVProcessor::rearrangeTextOrder(const SA_t* sa_array,
                       SA_t* psv, 
                       SA_t* nsv, 
                       const std::string& output_prefix,
                       size_t length,
                       const uint8_t* data) {
    const SA_t MAX_VAL = get_max_value<SA_t>();
    const SA_t MARK_BIT = (sizeof(SA_t) == 4) ? (1U << 31) : (1ULL << 63);
    
    #pragma omp parallel
    {
        #pragma omp for
        for(size_t i = 0; i < length; i++) {
            if(psv[i] != MAX_VAL) {
                psv[i] |= MARK_BIT;
            }
            if(nsv[i] != MAX_VAL) {
                nsv[i] |= MARK_BIT;
            }
        }

        #pragma omp for schedule(dynamic)
        for(size_t i = 0; i < length; i++) {

            if(psv[i] != MAX_VAL && (psv[i] & MARK_BIT)) {
                size_t curr_pos = i;
                SA_t curr_val = psv[i] & ~MARK_BIT;
                
                while(true) {
                    size_t next_pos = sa_array[curr_pos];
                    if(!(psv[next_pos] & MARK_BIT)) {  
                        break;
                    }
                    
                    SA_t next_val = psv[next_pos] & ~MARK_BIT;
                    psv[next_pos] = (curr_val == (MAX_VAL & ~MARK_BIT)) ? MAX_VAL : curr_val;
                    
                    if(next_pos == i) break;
                    
                    curr_pos = next_pos;
                    curr_val = next_val;
                }
            }

            if(nsv[i] != MAX_VAL && (nsv[i] & MARK_BIT)) {
                size_t curr_pos = i;
                SA_t curr_val = nsv[i] & ~MARK_BIT;
                
                while(true) {
                    size_t next_pos = sa_array[curr_pos];
                    if(!(nsv[next_pos] & MARK_BIT)) {
                        break;
                    }
                    
                    SA_t next_val = nsv[next_pos] & ~MARK_BIT;
                    nsv[next_pos] = (curr_val == (MAX_VAL & ~MARK_BIT)) ? MAX_VAL : curr_val;
                    
                    if(next_pos == i) break;
                    
                    curr_pos = next_pos;
                    curr_val = next_val;
                }
            }
        }

        #pragma omp for schedule(static)
        for(size_t i = 0; i < length; i++) {
            if(psv[i] != MAX_VAL && (psv[i] & MARK_BIT)) {
                psv[i] &= ~MARK_BIT;
            }
            if(nsv[i] != MAX_VAL && (nsv[i] & MARK_BIT)) {
                nsv[i] &= ~MARK_BIT;
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
}

template<typename SA_t>
void PipelinePSVNSVProcessor::processFullGPUWithGPUSA(SA_t* d_sa_array, const uint8_t* data, size_t length, const std::string& output_prefix) {
    profiler.start();

    const int block_size = DEFAULT_BLOCK_SIZE;
    const int num_blocks = (length + block_size - 1) / block_size;
    const size_t shared_mem_size = block_size * sizeof(SA_t);

    SA_t *d_psv_sa_order, *d_nsv_sa_order;
    SA_t *d_psv_text_order, *d_nsv_text_order;
    SA_t *d_block_mins; 
    cudaMalloc(&d_psv_sa_order, length * sizeof(SA_t));
    cudaMalloc(&d_nsv_sa_order, length * sizeof(SA_t));
    cudaMalloc(&d_psv_text_order, length * sizeof(SA_t));
    cudaMalloc(&d_nsv_text_order, length * sizeof(SA_t));
    cudaMalloc(&d_block_mins, num_blocks * sizeof(SA_t));

    std::vector<SA_t> h_psv_text_order(length);
    std::vector<SA_t> h_nsv_text_order(length);

    {
        cudaMemset(d_psv_sa_order, 0xFF, length * sizeof(SA_t));
        cudaMemset(d_nsv_sa_order, 0xFF, length * sizeof(SA_t)); 

        profiler.start();
        // Phase 1: Intra-block PSV/NSV computation (SA order)
        computePSVNSVKernel<<<num_blocks, block_size, shared_mem_size>>>(
            d_sa_array, d_psv_sa_order, d_nsv_sa_order, d_block_mins, length
        );
        profiler.stop("Phase 1: Intra-block computation");

        profiler.start();
        // Phase 2: Cross-block boundary processing (stay in SA order)
        processPSVNSVBoundariesKernel<<<num_blocks, block_size>>>(
            d_sa_array, d_psv_sa_order, d_nsv_sa_order,d_block_mins, length, block_size
        );
        profiler.stop("Phase 2: Cross-block processing");

        profiler.start();
        // Phase 3: Convert to text order using CUB sorting
        convertToTextOrderWithCUB(
            d_sa_array, d_psv_sa_order, d_nsv_sa_order,
            d_psv_text_order, d_nsv_text_order, length
        );
        profiler.stop("Phase 3: CUB text order conversion");

        cudaMemcpy(h_psv_text_order.data(), d_psv_text_order, length * sizeof(SA_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_nsv_text_order.data(), d_nsv_text_order, length * sizeof(SA_t), cudaMemcpyDeviceToHost);

        cudaFree(d_psv_sa_order);
        cudaFree(d_nsv_sa_order);
        cudaFree(d_psv_text_order);
        cudaFree(d_nsv_text_order);
        cudaFree(d_block_mins);
        cudaFree(d_sa_array);
    }

    profiler.stop("Full GPU Processing (Single Work Array)");

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA error occurred during GPU processing");
    }

    profiler.start();
    std::string lz_output = output_prefix + "_lz77.bin";
    ComputeLZ77(data, h_psv_text_order.data(), h_nsv_text_order.data(), length - 1, lz_output);
    profiler.stop("LZ77 Processing");
}

template<typename SA_t>
void PipelinePSVNSVProcessor::process(const SA_t* sa_array, const uint8_t* data, size_t length, const std::string& output_prefix) {
    try {
        std::cout << "Available GPU memory: " << available_memory << " bytes" << std::flush << std::endl;
        std::cout << "Input length: " << length << " bytes" << std::flush << std::endl;
        if (!canProcessFullGPU(length)) {
            std::cout << "Using stream processing mode" << std::flush << std::endl;
            processWithStreams(sa_array, data, length, output_prefix);
        }
    } catch (const std::bad_alloc& e) {
        std::cerr << "Memory allocation failed: " << e.what() << std::endl;
        throw;
    }
}

template<typename SA_t>
void PipelinePSVNSVProcessor::processWithStreams(const SA_t* sa_array, const uint8_t* data, size_t length, const std::string& output_prefix) {
    try {
        std::cout << "Starting stream processing..." << std::flush << std::endl;
        profiler.start();

        size_t max_batch_size = std::min(
            (available_memory / (3 * sizeof(SA_t))),
            static_cast<size_t>(256 * 1024 * 1024)
        );

        max_batch_size = (max_batch_size / DEFAULT_BLOCK_SIZE) * DEFAULT_BLOCK_SIZE;
        size_t optimal_batch_size = std::min(length, max_batch_size);
        size_t total_blocks = (length + optimal_batch_size - 1) / optimal_batch_size;

        std::cout << "Total blocks: " << total_blocks << std::endl;
        std::cout << "Final batch size: " << optimal_batch_size/(1024*1024) << "MB" << std::flush << std::endl;

        std::vector<SA_t> final_psv_results(length);
        std::vector<SA_t> final_nsv_results(length);
        std::vector<std::vector<SA_t>> all_block_mins;

        {
            cudaStream_t compute_stream;
            cudaStreamCreate(&compute_stream);

            const int block_size = DEFAULT_BLOCK_SIZE;
            const size_t shared_mem_size = block_size * sizeof(SA_t);

            SA_t* d_input, *d_psv_output, *d_nsv_output;
            cudaMalloc(&d_input, optimal_batch_size * sizeof(SA_t));
            cudaMalloc(&d_psv_output, optimal_batch_size * sizeof(SA_t));
            cudaMalloc(&d_nsv_output, optimal_batch_size * sizeof(SA_t));

            
            size_t processed_blocks = 0;
            for (size_t offset = 0; offset < length; offset += optimal_batch_size) {
                processed_blocks++;
                float progress = (processed_blocks * 100.0f) / total_blocks;
                std::cout << "\rGPU Processing: " << std::fixed << std::setprecision(2) 
                         << progress << "% [Block " << processed_blocks << "/" << total_blocks << "]" 
                         << std::flush;

                size_t current_batch = std::min(optimal_batch_size, length - offset);
                const int num_blocks = (current_batch + block_size - 1) / block_size;

                SA_t* d_block_mins;
                cudaMalloc(&d_block_mins, num_blocks * sizeof(SA_t));

                cudaMemcpyAsync(d_input, sa_array + offset, 
                            current_batch * sizeof(SA_t),
                            cudaMemcpyHostToDevice, 
                            compute_stream);

                computePSVNSVKernel<<<num_blocks, block_size, shared_mem_size, compute_stream>>>(
                    d_input, d_psv_output, d_nsv_output, d_block_mins, current_batch
                );

                cudaMemcpyAsync(&final_psv_results[offset], d_psv_output,
                              current_batch * sizeof(SA_t),
                              cudaMemcpyDeviceToHost,
                              compute_stream);
                cudaMemcpyAsync(&final_nsv_results[offset], d_nsv_output,
                              current_batch * sizeof(SA_t),
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
        
        #pragma omp parallel for schedule(dynamic)
        for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
            size_t block_start = block_idx * optimal_batch_size;
            size_t block_end = std::min(block_start + optimal_batch_size, length);
            
            if (block_idx > 0) {
                for (size_t i = block_start; i < block_end; ++i) {
                    if (is_invalid_value(final_psv_results[i])) {
                        block_psv_tasks[block_idx].push_back(i);
                    }
                }
            }

            if (block_idx < num_blocks - 1) {
                for (size_t i = block_start; i < block_end; ++i) {
                    if (is_invalid_value(final_nsv_results[i])) {
                        block_nsv_tasks[block_idx].push_back(i);
                    }
                }
            }
        }

        size_t total_tasks = 0;
        for (const auto& tasks : block_psv_tasks) total_tasks += tasks.size();
        for (const auto& tasks : block_nsv_tasks) total_tasks += tasks.size();
        
        std::cout << "Total merge tasks: " << total_tasks << std::endl;
        
        if (total_tasks > 0) {
            cudaStream_t task_stream;
            cudaStreamCreate(&task_stream);

            SA_t *d_sa_array, *d_results;
            size_t *d_positions;

            try {
                cudaMalloc(&d_sa_array, length * sizeof(SA_t));
                cudaMemcpyAsync(d_sa_array, sa_array, length * sizeof(SA_t),
                       cudaMemcpyHostToDevice, task_stream);

                const size_t MAX_BATCH_TASKS = 1024 * 1024;
                std::atomic<size_t> total_processed_tasks{0};

                // Process PSV tasks
                for (size_t block_idx = 0; block_idx < num_blocks; block_idx++) {
                    std::vector<size_t>& tasks = block_psv_tasks[block_idx];
                    if (tasks.empty()) continue;

                    std::cout << "\nProcessing PSV block " << block_idx + 1 << "/" << num_blocks 
                              << " with " << tasks.size() << " tasks" << std::endl;

                    for (size_t task_offset = 0; task_offset < tasks.size(); task_offset += MAX_BATCH_TASKS) {   
                        size_t current_batch_size = std::min(MAX_BATCH_TASKS, tasks.size() - task_offset);
                        
                        cudaMalloc(&d_positions, current_batch_size * sizeof(size_t));
                        cudaMalloc(&d_results, current_batch_size * sizeof(SA_t));

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

                        std::vector<SA_t> task_results(current_batch_size);
                        cudaMemcpyAsync(task_results.data(), d_results,
                                    current_batch_size * sizeof(SA_t),
                                    cudaMemcpyDeviceToHost, task_stream);
                        cudaStreamSynchronize(task_stream);

                        for (size_t i = 0; i < current_batch_size; ++i) {
                            if (!is_invalid_value(task_results[i])) {
                                final_psv_results[tasks[task_offset + i]] = task_results[i];
                            }
                        }

                        cudaFree(d_positions);
                        cudaFree(d_results);
                        d_positions = nullptr;
                        d_results = nullptr;

                        total_processed_tasks += current_batch_size;
                    }
                }

                // Process NSV tasks
                for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
                    std::vector<size_t>& tasks = block_nsv_tasks[block_idx];
                    if (tasks.empty()) continue;

                    std::cout << "\nProcessing NSV block " << block_idx + 1 << "/" << num_blocks 
                              << " with " << tasks.size() << " tasks" << std::endl;
                              
                    for (size_t task_offset = 0; task_offset < tasks.size(); task_offset += MAX_BATCH_TASKS) {
                        size_t current_batch_size = std::min(MAX_BATCH_TASKS, tasks.size() - task_offset);
        
                        cudaMalloc(&d_positions, current_batch_size * sizeof(size_t));
                        cudaMalloc(&d_results, current_batch_size * sizeof(SA_t));

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

                        std::vector<SA_t> task_results(current_batch_size);
                        cudaMemcpyAsync(task_results.data(), d_results,
                                    current_batch_size * sizeof(SA_t),
                                    cudaMemcpyDeviceToHost, task_stream);
                        cudaStreamSynchronize(task_stream);

                        for (size_t i = 0; i < current_batch_size; ++i) {
                            if (!is_invalid_value(task_results[i])) {
                                final_nsv_results[tasks[task_offset + i]] = task_results[i];
                            }
                        }

                        cudaFree(d_positions);
                        cudaFree(d_results);
                        d_positions = nullptr;
                        d_results = nullptr;

                        total_processed_tasks += current_batch_size;
                    }
                }
                
            } catch(const std::exception& e) {
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

// Explicit template instantiations
template void PipelinePSVNSVProcessor::process<uint32_t>(const uint32_t*, const uint8_t*, size_t, const std::string&);
template void PipelinePSVNSVProcessor::process<size_t>(const size_t*, const uint8_t*, size_t, const std::string&);

template void PipelinePSVNSVProcessor::processFullGPUWithGPUSA<uint32_t>(uint32_t*, const uint8_t*, size_t, const std::string&);
template void PipelinePSVNSVProcessor::processFullGPUWithGPUSA<size_t>(size_t*, const uint8_t*, size_t, const std::string&);

template void PipelinePSVNSVProcessor::processWithStreams<uint32_t>(const uint32_t*, const uint8_t*, size_t, const std::string&);
template void PipelinePSVNSVProcessor::processWithStreams<size_t>(const size_t*, const uint8_t*, size_t, const std::string&);

template void PipelinePSVNSVProcessor::rearrangeTextOrder<uint32_t>(const uint32_t*, uint32_t*, uint32_t*, const std::string&, size_t, const uint8_t*);
template void PipelinePSVNSVProcessor::rearrangeTextOrder<size_t>(const size_t*, size_t*, size_t*, const std::string&, size_t, const uint8_t*);

template std::pair<std::pair<size_t, size_t>, size_t> PipelinePSVNSVProcessor::LZFactor<uint32_t>(const uint8_t*, size_t, uint32_t, uint32_t, size_t);
template std::pair<std::pair<size_t, size_t>, size_t> PipelinePSVNSVProcessor::LZFactor<size_t>(const uint8_t*, size_t, size_t, size_t, size_t);

template void PipelinePSVNSVProcessor::ComputeLZ77<uint32_t>(const uint8_t*, uint32_t*, uint32_t*, size_t, std::string);
template void PipelinePSVNSVProcessor::ComputeLZ77<size_t>(const uint8_t*, size_t*, size_t*, size_t, std::string);

// Explicit kernel instantiations
template __global__ void processPSVTasksKernel<uint32_t>(const uint32_t*, uint32_t*, const size_t*, const size_t, const size_t, const size_t);
template __global__ void processPSVTasksKernel<size_t>(const size_t*, size_t*, const size_t*, const size_t, const size_t, const size_t);

template __global__ void processNSVTasksKernel<uint32_t>(const uint32_t*, uint32_t*, const size_t*, const size_t, const size_t, const size_t);
template __global__ void processNSVTasksKernel<size_t>(const size_t*, size_t*, const size_t*, const size_t, const size_t, const size_t);

template __global__ void computePSVNSVKernel<uint32_t>(const uint32_t*, uint32_t*, uint32_t*, uint32_t*, const size_t);
template __global__ void computePSVNSVKernel<size_t>(const size_t*, size_t*, size_t*, size_t*, const size_t);

template __global__ void processPSVNSVBoundariesKernel<uint32_t>(const uint32_t*, uint32_t*, uint32_t*, const uint32_t*, const size_t, const size_t);
template __global__ void processPSVNSVBoundariesKernel<size_t>(const size_t*, size_t*, size_t*, const size_t*, const size_t, const size_t);
template void PipelinePSVNSVProcessor::convertToTextOrderWithCUB<uint32_t>(
    const uint32_t*, const uint32_t*, const uint32_t*, uint32_t*, uint32_t*, size_t);
template void PipelinePSVNSVProcessor::convertToTextOrderWithCUB<size_t>(
    const size_t*, const size_t*, const size_t*, size_t*, size_t*, size_t);