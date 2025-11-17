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
#include <stack>
#include <unordered_set>
#include <stdexcept>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

// Boundary search window size: 0 = unlimited, N = limit to N elements
#ifndef BOUNDARY_SEARCH_WINDOW
#define BOUNDARY_SEARCH_WINDOW 0
#endif

// Safety guard: upper bound on how many unresolved indices we keep during stream mode
constexpr size_t UNFOUND_INDEX_THRESHOLD = 20ULL * 1000ULL * 1000ULL;  // 20 million
// Force stream mode to create at least this many chunks when length >> available GPU memory
constexpr size_t MIN_STREAM_CHUNKS = 4;

inline size_t countTrailingZeros64(uint64_t value) {
#if defined(_MSC_VER)
    unsigned long index;
    _BitScanForward64(&index, value);
    return static_cast<size_t>(index);
#else
    return static_cast<size_t>(__builtin_ctzll(value));
#endif
}

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

    // PSV: search backwards from pos-1, bounded by [start_pos, end_pos)
    size_t search_end = (pos > start_pos) ? pos : start_pos;
    for (size_t j = search_end; j-- > start_pos;) {
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

    // NSV: search forwards from pos+1, bounded by [start_pos, end_pos)
    size_t search_start = (pos + 1 > start_pos) ? (pos + 1) : start_pos;
    for (size_t j = search_start; j < end_pos; ++j) {
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

    // Load to shared memory (boundary threads load MAX_VAL)
    shared_data[tid] = (gid < length) ? input[gid] : MAX_VAL;
    __syncthreads();

    if (gid < length) {
        const SA_t current = shared_data[tid];

        // PSV: search backwards in block
        SA_t psv_val = MAX_VAL;
        for(int i = tid - 1; i >= 0; --i) {
            if(shared_data[i] < current) {
                psv_val = shared_data[i];
                break;
            }
        }
        psv_output[gid] = psv_val;

        // NSV: search forwards in block (no need to check i < length, shared_data already padded)
        SA_t nsv_val = MAX_VAL;
        for(int i = tid + 1; i < blockDim.x; ++i) {
            if(shared_data[i] < current) {
                nsv_val = shared_data[i];
                break;
            }
        }
        nsv_output[gid] = nsv_val;
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

    // PSV search with configurable window
    if (psv_val == MAX_VAL && bid > 0) {

        int search_block = bid - 1;
        while (search_block >= 0 && block_min_output[search_block] >= current) {
            search_block--;
        }


        if (search_block >= 0) {
            size_t search_end = min((size_t)gid, (size_t)(search_block + 1) * block_size);
#if BOUNDARY_SEARCH_WINDOW == 0
            size_t search_start = 0;  // Unlimited search
#else
            size_t search_start = max((size_t)0, search_end > BOUNDARY_SEARCH_WINDOW ? search_end - BOUNDARY_SEARCH_WINDOW : 0);
#endif

            for (size_t i = search_end - 1; i >= search_start && psv_val == MAX_VAL; --i) {
                if (sa_array[i] < current) {
                    psv_val = sa_array[i];
                    break;
                }
                if (i == 0) break;
            }
        }
    }
    
    // NSV search with configurable window
    if (nsv_val == MAX_VAL && bid + 1 < total_blocks) {

        size_t search_block = bid + 1;
        while (search_block < total_blocks && block_min_output[search_block] >= current) {
            search_block++;
        }

        if (search_block < total_blocks) {
            size_t search_start = max((size_t)gid + 1, search_block * block_size);
#if BOUNDARY_SEARCH_WINDOW == 0
            size_t search_end = length;  // Unlimited search
#else
            size_t search_end = min(length, search_start + BOUNDARY_SEARCH_WINDOW);
#endif

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

void PipelinePSVNSVProcessor::calculateAvailableMemory() {
    size_t free_memory, total_memory;
    cudaMemGetInfo(&free_memory, &total_memory);
    available_memory = static_cast<size_t>(free_memory * MEMORY_RESERVE_RATIO);
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
    std::ofstream out_file(file_name, std::ios::binary);

    if (!out_file) {
        throw std::runtime_error("Failed to open LZ77 output file");
    }

    while(i < n) {
        SA_t psv = d_psv_text[i];
        SA_t nsv = d_nsv_text[i];
        
        auto result = LZFactor(data, i, psv, nsv, n);
        size_t pos = result.first.first;
        size_t len = result.first.second;
        i = result.second;
    
        out_file.write(reinterpret_cast<const char*>(&pos), sizeof(size_t));
        out_file.write(reinterpret_cast<const char*>(&len), sizeof(size_t));
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

    profiler.start();

    // Allocate single temp buffer (1n) for scatter operations
    size_t temp_size = length * sizeof(SA_t);
    std::cout << "  Allocating temp buffer: " << temp_size / (1024.0 * 1024.0) << " MB" << std::endl;
    std::vector<SA_t> temp_buffer(length);

    // Scatter PSV: temp[sa[i]] = psv[i]
    #pragma omp parallel for schedule(static)
    for(size_t i = 0; i < length; i++) {
        size_t text_pos = sa_array[i];
        temp_buffer[text_pos] = psv[i];
    }

    // Copy back (sequential, cache-friendly)
    std::memcpy(psv, temp_buffer.data(), temp_size);

    // Scatter NSV: reuse temp_buffer
    #pragma omp parallel for schedule(static)
    for(size_t i = 0; i < length; i++) {
        size_t text_pos = sa_array[i];
        temp_buffer[text_pos] = nsv[i];
    }

    // Copy back
    std::memcpy(nsv, temp_buffer.data(), temp_size);

    profiler.stop("SA-order to Text-order conversion");

    profiler.start();
    std::string lz_output = output_prefix + "_lz77.bin";
    ComputeLZ77(data, psv, nsv, length - 1, lz_output);
    profiler.stop("LZ77 Processing");
}

template<typename SA_t>
void PipelinePSVNSVProcessor::rearrangeTextOrderInPlace(const SA_t* sa_array,
                       SA_t* psv,
                       SA_t* nsv,
                       const std::string& output_prefix,
                       size_t length,
                       const uint8_t* data) {

    // Memory-optimized version using fused cycle-following algorithm
    // Peak memory: 3n×sizeof(SA_t) + n/8 (vs 4n×sizeof(SA_t) for temp buffer version)
    //
    // Key optimization: Process PSV and NSV simultaneously in a single pass
    // - Both arrays share the same cycle structure (defined by SA)
    // - Reduces memory accesses by ~33% compared to separate processing
    // - Better cache locality by accessing PSV[i] and NSV[i] together

    size_t bitvector_size = (length + 7) / 8;  // Round up to byte boundary
    std::cout << "\n=== In-Place Text-Order Conversion (Memory-Optimized) ===" << std::endl;
    std::cout << "  Method: Fused cycle-following (PSV+NSV)" << std::endl;
    std::cout << "  Bitvector overhead: " << bitvector_size / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "  Memory saved vs temp buffer: " << (length * sizeof(SA_t)) / (1024.0 * 1024.0) << " MB" << std::endl;

    // Bitvector to track visited positions (n/8 bytes)
    std::vector<bool> visited(length, false);

    profiler.start();

    // ======== Fused Rearrangement: Process PSV and NSV simultaneously ========
    // Since PSV and NSV share the same permutation structure (defined by SA),
    // we can follow each cycle once and update both arrays together.
    // This reduces the number of passes over the data from 2 to 1.

    for(size_t sa_pos = 0; sa_pos < length; sa_pos++) {
        if (visited[sa_pos]) continue;  // Already processed in a previous cycle

        // We want to perform: psv_text[sa[i]] = psv_sa[i]
        // Start from SA position sa_pos, save its value
        size_t current_sa = sa_pos;
        SA_t temp_psv = psv[current_sa];
        SA_t temp_nsv = nsv[current_sa];

        // Follow the cycle: write psv_sa[current_sa] to position sa[current_sa]
        size_t text_pos = sa_array[current_sa];
        while(text_pos != sa_pos) {
            visited[current_sa] = true;

            // Write current values to text position, then read from text position for next iteration
            SA_t next_psv = psv[text_pos];
            SA_t next_nsv = nsv[text_pos];

            psv[text_pos] = temp_psv;
            nsv[text_pos] = temp_nsv;

            // Move to next position in cycle
            temp_psv = next_psv;
            temp_nsv = next_nsv;
            current_sa = text_pos;
            text_pos = sa_array[current_sa];
        }

        // Close the cycle: write the saved values to the starting position
        visited[current_sa] = true;
        psv[text_pos] = temp_psv;
        nsv[text_pos] = temp_nsv;
    }

    profiler.stop("Fused PSV+NSV in-place rearrangement");

    // Note: ComputeLZ77 is NOT called here anymore
    // Caller is responsible for calling ComputeLZ77 after releasing SA to reduce peak memory

    /* ======== Alternative: Separate Processing (Kept for Reference) ========
     * This version processes PSV and NSV in two separate passes.
     * Pros: Easier to understand and debug
     * Cons: ~33% more memory accesses, worse cache performance
     *
     * // Phase 1: Rearrange PSV
     * for(size_t start = 0; start < length; start++) {
     *     if (visited[start]) continue;
     *     size_t current = start;
     *     SA_t temp = psv[start];
     *     size_t next = sa_array[current];
     *     while(next != start) {
     *         visited[current] = true;
     *         psv[current] = psv[next];
     *         current = next;
     *         next = sa_array[current];
     *     }
     *     visited[current] = true;
     *     psv[current] = temp;
     * }
     *
     * // Reset visited
     * std::fill(visited.begin(), visited.end(), false);
     *
     * // Phase 2: Rearrange NSV (same logic)
     * for(size_t start = 0; start < length; start++) {
     *     if (visited[start]) continue;
     *     size_t current = start;
     *     SA_t temp = nsv[start];
     *     size_t next = sa_array[current];
     *     while(next != start) {
     *         visited[current] = true;
     *         nsv[current] = nsv[next];
     *         current = next;
     *         next = sa_array[current];
     *     }
     *     visited[current] = true;
     *     nsv[current] = temp;
     * }
     */
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

    // Memory profiling: Log initial state
    size_t free_mem_start, total_mem;
    cudaMemGetInfo(&free_mem_start, &total_mem);
    std::cout << "\n=== Full GPU Memory Profile ===" << std::endl;
    std::cout << "GPU Total Memory: " << total_mem / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "GPU Free Memory (start): " << free_mem_start / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Input SA already on GPU: " << length * sizeof(SA_t) / (1024.0 * 1024.0) << " MB" << std::endl;

    SA_t *d_psv_sa_order, *d_nsv_sa_order;
    SA_t *d_block_mins;
    SA_t *d_temp;

    // Allocate all buffers upfront in single contiguous block (4n total)
    // Layout: [psv(1n) | nsv(1n) | temp(1n) | block_mins(~0.001n)]
    // d_temp doubles as block_mins storage during Phase 1-2 (first num_blocks elements)
    size_t total_size = (3 * length + num_blocks) * sizeof(SA_t);
    SA_t* d_combined;
    cudaMalloc(&d_combined, total_size);
    d_psv_sa_order = d_combined;
    d_nsv_sa_order = d_combined + length;
    d_temp = d_combined + 2 * length;
    d_block_mins = d_temp;  // Reuse first num_blocks elements of d_temp

    std::cout << "Allocated combined buffer: " << total_size / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "  - d_psv_sa_order: " << length * sizeof(SA_t) / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "  - d_nsv_sa_order: " << length * sizeof(SA_t) / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "  - d_temp (shared with block_mins): " << length * sizeof(SA_t) / (1024.0 * 1024.0)
              << " MB (first " << num_blocks << " blocks = " << num_blocks * sizeof(SA_t) / 1024.0 << " KB)" << std::endl;

    size_t free_mem_after_alloc;
    cudaMemGetInfo(&free_mem_after_alloc, &total_mem);
    std::cout << "GPU Free Memory (after alloc): " << free_mem_after_alloc / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Peak GPU Memory Used: " << (free_mem_start - free_mem_after_alloc) / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "================================\n" << std::endl;

    std::vector<SA_t> h_psv_text_order(length);
    std::vector<SA_t> h_nsv_text_order(length);

    {
        // No longer need cudaMemset - kernel writes all values directly

        profiler.start();
        // Phase 1+2: Pipelined PSV/NSV computation (no sync between kernels)
        computePSVNSVKernel<<<num_blocks, block_size, shared_mem_size>>>(
            d_sa_array, d_psv_sa_order, d_nsv_sa_order, d_block_mins, length
        );

        processPSVNSVBoundariesKernel<<<num_blocks, block_size>>>(
            d_sa_array, d_psv_sa_order, d_nsv_sa_order, d_block_mins, length, block_size
        );
        cudaDeviceSynchronize(); // Single sync point for both kernels
        profiler.stop("Phase 1+2: PSV/NSV computation (pipelined)");

        profiler.start();
        // Phase 3: Reuse d_temp for scatter (already allocated, shared with block_mins)
        // d_temp was reused as block_mins in Phase 1-2, now use full buffer for scatter
        auto sa_ptr = thrust::device_pointer_cast(d_sa_array);
        auto temp_ptr = thrust::device_pointer_cast(d_temp);
        auto psv_ptr = thrust::device_pointer_cast(d_psv_sa_order);
        auto nsv_ptr = thrust::device_pointer_cast(d_nsv_sa_order);

        // Process PSV: scatter to temp, copy back
        thrust::scatter(psv_ptr, psv_ptr + length, sa_ptr, temp_ptr);
        thrust::copy(temp_ptr, temp_ptr + length, psv_ptr);

        // Process NSV: reuse same temp buffer
        thrust::scatter(nsv_ptr, nsv_ptr + length, sa_ptr, temp_ptr);
        thrust::copy(temp_ptr, temp_ptr + length, nsv_ptr);

        profiler.stop("Phase 3: Text order conversion (4n peak - optimal)");

        cudaMemcpy(h_psv_text_order.data(), d_psv_sa_order, length * sizeof(SA_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_nsv_text_order.data(), d_nsv_sa_order, length * sizeof(SA_t), cudaMemcpyDeviceToHost);

        cudaFree(d_combined);  // Free single combined allocation
        // Note: d_sa_array is NOT freed here - caller is responsible for freeing it
    }

    size_t free_mem_end;
    cudaMemGetInfo(&free_mem_end, &total_mem);
    std::cout << "\nGPU Free Memory (after cleanup): " << free_mem_end / (1024.0 * 1024.0) << " MB\n" << std::endl;

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


// Optimized PSV resolution using targeted search with block_min pruning
// Only searches unfound positions, uses block_min to skip impossible blocks
// Time: O(unfound_count * avg_blocks_searched), Space: O(1)
template<typename SA_t>
void PipelinePSVNSVProcessor::resolvePSVWithPruning(
    const SA_t* sa_array,
    SA_t* psv,
    const std::vector<size_t>& unfound_indices,
    const std::vector<SA_t>& block_mins,
    size_t length,
    size_t block_size
) {
    if (unfound_indices.empty()) return;

    const SA_t MAX_VAL = get_max_value<SA_t>();
    size_t blocks_searched = 0;
    size_t blocks_skipped = 0;

    #pragma omp parallel for reduction(+:blocks_searched,blocks_skipped)
    for (size_t idx = 0; idx < unfound_indices.size(); ++idx) {
        size_t pos = unfound_indices[idx];
        SA_t current = sa_array[pos];
        size_t current_block = pos / block_size;

        // Search backwards through blocks
        for (int block = static_cast<int>(current_block) - 1; block >= 0; --block) {
            // Use block_min to prune: if minimum value in block >= current, skip
            if (block_mins[block] >= current) {
                blocks_skipped++;
                continue;
            }

            // Search this block from right to left
            size_t block_start = block * block_size;
            size_t block_end = std::min((block + 1) * block_size, length);

            blocks_searched++;
            for (size_t i = std::min(pos, block_end) - 1; i >= block_start; --i) {
                if (sa_array[i] < current) {
                    psv[pos] = sa_array[i];
                    goto found_psv;  // Found PSV, move to next unfound position
                }
                if (i == 0) break;  // Avoid underflow
            }
        }

        // No PSV found
        psv[pos] = MAX_VAL;

        found_psv:;
    }

    std::cout << "  PSV resolved " << unfound_indices.size() << " positions "
              << "(searched: " << blocks_searched << " blocks, skipped: " << blocks_skipped << " blocks)" << std::endl;
}

// Optimized NSV resolution using targeted search with block_min pruning
// Only searches unfound positions, uses block_min to skip impossible blocks
// Time: O(unfound_count * avg_blocks_searched), Space: O(1)
template<typename SA_t>
void PipelinePSVNSVProcessor::resolveNSVWithPruning(
    const SA_t* sa_array,
    SA_t* nsv,
    const std::vector<size_t>& unfound_indices,
    const std::vector<SA_t>& block_mins,
    size_t length,
    size_t block_size
) {
    if (unfound_indices.empty()) return;

    const SA_t MAX_VAL = get_max_value<SA_t>();
    size_t total_blocks = (length + block_size - 1) / block_size;
    size_t blocks_searched = 0;
    size_t blocks_skipped = 0;

    #pragma omp parallel for reduction(+:blocks_searched,blocks_skipped)
    for (size_t idx = 0; idx < unfound_indices.size(); ++idx) {
        size_t pos = unfound_indices[idx];
        SA_t current = sa_array[pos];
        size_t current_block = pos / block_size;

        // Search forwards through blocks
        for (size_t block = current_block + 1; block < total_blocks; ++block) {
            // Use block_min to prune: if minimum value in block >= current, skip
            if (block_mins[block] >= current) {
                blocks_skipped++;
                continue;
            }

            // Search this block from left to right
            size_t block_start = block * block_size;
            size_t block_end = std::min((block + 1) * block_size, length);

            blocks_searched++;
            for (size_t i = std::max(pos + 1, block_start); i < block_end; ++i) {
                if (sa_array[i] < current) {
                    nsv[pos] = sa_array[i];
                    goto found_nsv;  // Found NSV, move to next unfound position
                }
            }
        }

        // No NSV found
        nsv[pos] = MAX_VAL;

        found_nsv:;
    }

    std::cout << "  NSV resolved " << unfound_indices.size() << " positions "
              << "(searched: " << blocks_searched << " blocks, skipped: " << blocks_skipped << " blocks)" << std::endl;
}

template<typename SA_t>
void PipelinePSVNSVProcessor::processWithStreams(std::vector<SA_t>& sa_array, const uint8_t* data, size_t length, const std::string& output_prefix) {
    const SA_t* sa_ptr = sa_array.data();  // Cache pointer before potential reallocation
    try {
        std::cout << "\n=== Starting Optimized Stream Processing ===" << std::endl;
        profiler.start();

        // Calculate optimal chunk size based on available GPU memory
        // Each chunk needs 3 buffers: input + psv + nsv (plus small block_mins)
        size_t max_chunk_size = available_memory / (3 * sizeof(SA_t));

        // Align to block size for efficient GPU processing
        max_chunk_size = (max_chunk_size / DEFAULT_BLOCK_SIZE) * DEFAULT_BLOCK_SIZE;

        // Ensure minimum chunk size to avoid divide-by-zero
        if (max_chunk_size == 0) {
            throw std::runtime_error(
                "Insufficient GPU memory for stream processing. "
                "Available memory: " + std::to_string(available_memory / (1024.0 * 1024.0)) + " MB, "
                "Minimum required: " + std::to_string((3 * DEFAULT_BLOCK_SIZE * sizeof(SA_t)) / (1024.0 * 1024.0)) + " MB"
            );
        }

        // Don't exceed file length; optionally split into more chunks to reduce inter-chunk dependencies
        size_t chunk_size = std::min(length, max_chunk_size);
        if (length > chunk_size) {
            size_t desired_chunks = (length + chunk_size - 1) / chunk_size;
            desired_chunks = std::max(desired_chunks, MIN_STREAM_CHUNKS);
            chunk_size = (length + desired_chunks - 1) / desired_chunks;
            chunk_size = (chunk_size / DEFAULT_BLOCK_SIZE) * DEFAULT_BLOCK_SIZE;
            if (chunk_size == 0) {
                chunk_size = DEFAULT_BLOCK_SIZE;
            }
        }
        size_t num_chunks = (length + chunk_size - 1) / chunk_size;

        std::cout << "\nConfiguration:" << std::endl;
        std::cout << "  Total chunks: " << num_chunks << std::endl;
        std::cout << "  Chunk size: " << chunk_size / (1024 * 1024) << " MB" << std::endl;

        // Allocate result arrays
        size_t cpu_alloc_size = 2 * length * sizeof(SA_t);
        std::cout << "\nCPU Memory Allocation:" << std::endl;
        std::cout << "  PSV + NSV results: " << cpu_alloc_size / (1024.0 * 1024.0) << " MB" << std::endl;
        std::vector<SA_t> psv_results(length, get_max_value<SA_t>());
        std::vector<SA_t> nsv_results(length, get_max_value<SA_t>());

        size_t sa_memory = sa_array.size() * sizeof(SA_t);
        size_t input_memory = length * sizeof(uint8_t);
        double estimated_peak_mb = (cpu_alloc_size + sa_memory + input_memory) / (1024.0 * 1024.0);
        std::cout << "  Input data (resident in main): " << input_memory / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "  SA (CPU): " << sa_memory / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "  Estimated CPU peak before metadata: " << estimated_peak_mb
                  << " MB (data + SA + PSV + NSV)" << std::endl;

        // Metadata for unfound positions and block minimums
        std::vector<size_t> global_psv_unfound;
        std::vector<size_t> global_nsv_unfound;
        size_t unfound_bitmap_words = (length + 63) / 64;
        std::vector<uint64_t> psv_unfound_bitmap(unfound_bitmap_words, 0);
        std::vector<uint64_t> nsv_unfound_bitmap(unfound_bitmap_words, 0);
        size_t psv_unfound_count = 0;
        size_t nsv_unfound_count = 0;
        const int block_size = DEFAULT_BLOCK_SIZE;
        size_t total_blocks = (length + block_size - 1) / block_size;
        size_t block_mins_size = total_blocks * sizeof(SA_t);
        std::cout << "  Block mins metadata: " << block_mins_size / (1024.0 * 1024.0) << " MB" << std::endl;
        std::vector<SA_t> global_block_mins(total_blocks);
        std::cout << "  Estimated CPU peak with metadata: "
                  << (estimated_peak_mb + block_mins_size / (1024.0 * 1024.0))
                  << " MB (actual peak may vary with unfound indices)" << std::endl;

        // ======== PHASE 1: GPU Chunk Processing + Metadata Collection ========
        std::cout << "\n=== Phase 1: GPU Chunk Processing ===" << std::endl;
        size_t free_before, total_gpu;
        cudaMemGetInfo(&free_before, &total_gpu);
        std::cout << "GPU Memory before Phase 1: " << free_before / (1024.0 * 1024.0) << " MB free / "
                  << total_gpu / (1024.0 * 1024.0) << " MB total" << std::endl;

        profiler.start();
        {
            cudaStream_t compute_stream;
            cudaStreamCreate(&compute_stream);

            const size_t shared_mem_size = block_size * sizeof(SA_t);

            // Allocate GPU buffers for one chunk
            size_t gpu_chunk_alloc = 3 * chunk_size * sizeof(SA_t);
            std::cout << "GPU per-chunk allocation: " << gpu_chunk_alloc / (1024.0 * 1024.0) << " MB "
                      << "(input + psv + nsv)" << std::endl;

            SA_t *d_input, *d_psv_output, *d_nsv_output, *d_block_mins;
            cudaMalloc(&d_input, chunk_size * sizeof(SA_t));
            cudaMalloc(&d_psv_output, chunk_size * sizeof(SA_t));
            cudaMalloc(&d_nsv_output, chunk_size * sizeof(SA_t));

            for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
                size_t offset = chunk_idx * chunk_size;
                size_t current_chunk_size = std::min(chunk_size, length - offset);
                const int num_blocks = (current_chunk_size + block_size - 1) / block_size;

                float progress = ((chunk_idx + 1) * 100.0f) / num_chunks;
                std::cout << "\rPhase 1 - GPU Processing: " << std::fixed << std::setprecision(1)
                         << progress << "% [" << (chunk_idx + 1) << "/" << num_chunks << "]" << std::flush;

                cudaMalloc(&d_block_mins, num_blocks * sizeof(SA_t));

                // Upload chunk to GPU
                cudaMemcpyAsync(d_input, sa_ptr + offset,
                              current_chunk_size * sizeof(SA_t),
                              cudaMemcpyHostToDevice,
                              compute_stream);

                // Step 1: Compute PSV/NSV within blocks
                computePSVNSVKernel<<<num_blocks, block_size, shared_mem_size, compute_stream>>>(
                    d_input, d_psv_output, d_nsv_output, d_block_mins, current_chunk_size
                );

                // Step 2: Process cross-block boundaries within this chunk (GPU merge)
                processPSVNSVBoundariesKernel<<<num_blocks, block_size, 0, compute_stream>>>(
                    d_input, d_psv_output, d_nsv_output, d_block_mins,
                    current_chunk_size, block_size
                );

                // Download results and block_mins
                cudaMemcpyAsync(&psv_results[offset], d_psv_output,
                              current_chunk_size * sizeof(SA_t),
                              cudaMemcpyDeviceToHost, compute_stream);
                cudaMemcpyAsync(&nsv_results[offset], d_nsv_output,
                              current_chunk_size * sizeof(SA_t),
                              cudaMemcpyDeviceToHost, compute_stream);

                // Save block_mins for this chunk to global array
                size_t block_offset = (offset / block_size);
                cudaMemcpyAsync(&global_block_mins[block_offset], d_block_mins,
                              num_blocks * sizeof(SA_t),
                              cudaMemcpyDeviceToHost, compute_stream);

                cudaStreamSynchronize(compute_stream);
                cudaFree(d_block_mins);

                // Collect metadata: unfound positions in this chunk
                // After GPU boundary merge, these should only be inter-chunk boundaries
                for (size_t i = offset; i < offset + current_chunk_size; ++i) {
                    if (is_invalid_value(psv_results[i])) {
                        size_t word = i >> 6;
                        size_t bit = i & 63;
                        psv_unfound_bitmap[word] |= (1ULL << bit);
                        ++psv_unfound_count;
                        if (psv_unfound_count > UNFOUND_INDEX_THRESHOLD) {
                            throw std::runtime_error(
                                "PSV unfound indices exceeded threshold ("
                                + std::to_string(UNFOUND_INDEX_THRESHOLD)
                                + ") during Phase 1. Reduce chunk size or adjust threshold.");
                        }
                    }
                    if (is_invalid_value(nsv_results[i])) {
                        size_t word = i >> 6;
                        size_t bit = i & 63;
                        nsv_unfound_bitmap[word] |= (1ULL << bit);
                        ++nsv_unfound_count;
                        if (nsv_unfound_count > UNFOUND_INDEX_THRESHOLD) {
                            throw std::runtime_error(
                                "NSV unfound indices exceeded threshold ("
                                + std::to_string(UNFOUND_INDEX_THRESHOLD)
                                + ") during Phase 1. Reduce chunk size or adjust threshold.");
                        }
                    }
                }
            }

            std::cout << std::endl;
            cudaStreamDestroy(compute_stream);
            cudaFree(d_input);
            cudaFree(d_psv_output);
            cudaFree(d_nsv_output);
        }

        profiler.stop("Phase 1: GPU Chunk Processing");

        auto populate_unfound_indices = [&](const std::vector<uint64_t>& bitmap,
                                            size_t count,
                                            std::vector<size_t>& output) {
            output.clear();
            output.reserve(count);
            for (size_t word_idx = 0; word_idx < bitmap.size(); ++word_idx) {
                uint64_t word = bitmap[word_idx];
                while (word) {
                    size_t bit = countTrailingZeros64(word);
                    size_t pos = word_idx * 64 + bit;
                    if (pos < length) {
                        output.push_back(pos);
                    }
                    word &= (word - 1);
                }
            }
        };

        if (psv_unfound_count > 0) {
            populate_unfound_indices(psv_unfound_bitmap, psv_unfound_count, global_psv_unfound);
        }
        if (nsv_unfound_count > 0) {
            populate_unfound_indices(nsv_unfound_bitmap, nsv_unfound_count, global_nsv_unfound);
        }
        psv_unfound_bitmap.clear();
        psv_unfound_bitmap.shrink_to_fit();
        nsv_unfound_bitmap.clear();
        nsv_unfound_bitmap.shrink_to_fit();

        size_t free_after_phase1;
        cudaMemGetInfo(&free_after_phase1, &total_gpu);
        std::cout << "GPU Memory after Phase 1: " << free_after_phase1 / (1024.0 * 1024.0) << " MB free" << std::endl;

        std::cout << "\nMetadata collected (after GPU intra-chunk merge):" << std::endl;
        std::cout << "  PSV unfound (inter-chunk boundaries): " << global_psv_unfound.size() << " positions ("
                  << (global_psv_unfound.size() * 100.0 / length) << "%)" << std::endl;
        std::cout << "  NSV unfound (inter-chunk boundaries): " << global_nsv_unfound.size() << " positions ("
                  << (global_nsv_unfound.size() * 100.0 / length) << "%)" << std::endl;

        if (num_chunks > 1) {
            std::cout << "  Note: GPU handled intra-chunk boundaries, CPU will handle "
                      << num_chunks - 1 << " inter-chunk boundaries" << std::endl;
        }

        // ======== PHASE 2: CPU Targeted Search with Block Min Pruning ========
        std::cout << "\n=== Phase 2: CPU Targeted Search ===" << std::endl;
        profiler.start();

        // Resolve PSV using targeted search with block_min pruning
        if (!global_psv_unfound.empty()) {
            resolvePSVWithPruning(sa_ptr, psv_results.data(), global_psv_unfound,
                                 global_block_mins, length, block_size);
        }

        // Resolve NSV using targeted search with block_min pruning
        if (!global_nsv_unfound.empty()) {
            resolveNSVWithPruning(sa_ptr, nsv_results.data(), global_nsv_unfound,
                                 global_block_mins, length, block_size);
        }

        // free metadata vectors
        global_psv_unfound.clear();
        global_psv_unfound.shrink_to_fit();
        global_nsv_unfound.clear();
        global_nsv_unfound.shrink_to_fit();

        profiler.stop("Phase 2: CPU Targeted Search");

        // ======== PHASE 3: Text Order Conversion ========
        std::cout << "\n=== Phase 3: SA-order to Text-order Conversion ===" << std::endl;

        // Convert to text order (PSV/NSV from SA-order to text-order)
        // Note: rearrangeTextOrderInPlace no longer calls ComputeLZ77
        rearrangeTextOrderInPlace(sa_ptr, psv_results.data(), nsv_results.data(), output_prefix, length, data);

        // ======== SA Cleanup ========
        // Critical: Free SA immediately after conversion to reduce peak memory
        // Peak memory before: SA + PSV + NSV + data + metadata
        // Peak memory after: PSV + NSV + data + metadata (saves ~57.6 GB for 7.2GB input)
        size_t sa_memory_mb = (sa_array.size() * sizeof(SA_t)) / (1024.0 * 1024.0);
        std::cout << "\nReleasing SA (freeing " << sa_memory_mb << " MB)" << std::endl;
        sa_array.clear();
        sa_array.shrink_to_fit();

        // ======== PHASE 4: LZ77 Factorization ========
        std::cout << "\n=== Phase 4: LZ77 Factorization (SA already freed) ===" << std::endl;
        profiler.start();
        std::string lz_output = output_prefix + "_lz77.bin";
        ComputeLZ77(data, psv_results.data(), nsv_results.data(), length - 1, lz_output);
        profiler.stop("LZ77 Processing");

        std::cout << "\n=== Stream Processing Complete ===" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error in stream processing: " << e.what() << std::endl;
        throw;
    }
}

// Explicit template instantiations
template void PipelinePSVNSVProcessor::processFullGPUWithGPUSA<uint32_t>(uint32_t*, const uint8_t*, size_t, const std::string&);
template void PipelinePSVNSVProcessor::processFullGPUWithGPUSA<size_t>(size_t*, const uint8_t*, size_t, const std::string&);

template void PipelinePSVNSVProcessor::resolvePSVWithPruning<uint32_t>(const uint32_t*, uint32_t*, const std::vector<size_t>&, const std::vector<uint32_t>&, size_t, size_t);
template void PipelinePSVNSVProcessor::resolvePSVWithPruning<size_t>(const size_t*, size_t*, const std::vector<size_t>&, const std::vector<size_t>&, size_t, size_t);

template void PipelinePSVNSVProcessor::resolveNSVWithPruning<uint32_t>(const uint32_t*, uint32_t*, const std::vector<size_t>&, const std::vector<uint32_t>&, size_t, size_t);
template void PipelinePSVNSVProcessor::resolveNSVWithPruning<size_t>(const size_t*, size_t*, const std::vector<size_t>&, const std::vector<size_t>&, size_t, size_t);

template void PipelinePSVNSVProcessor::processWithStreams<uint32_t>(std::vector<uint32_t>&, const uint8_t*, size_t, const std::string&);
template void PipelinePSVNSVProcessor::processWithStreams<size_t>(std::vector<size_t>&, const uint8_t*, size_t, const std::string&);

template void PipelinePSVNSVProcessor::rearrangeTextOrder<uint32_t>(const uint32_t*, uint32_t*, uint32_t*, const std::string&, size_t, const uint8_t*);
template void PipelinePSVNSVProcessor::rearrangeTextOrder<size_t>(const size_t*, size_t*, size_t*, const std::string&, size_t, const uint8_t*);

template void PipelinePSVNSVProcessor::rearrangeTextOrderInPlace<uint32_t>(const uint32_t*, uint32_t*, uint32_t*, const std::string&, size_t, const uint8_t*);
template void PipelinePSVNSVProcessor::rearrangeTextOrderInPlace<size_t>(const size_t*, size_t*, size_t*, const std::string&, size_t, const uint8_t*);

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
