#include "LZ77_processor.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
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
#include "logger.cuh"
#include <spdlog/fmt/fmt.h>
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
    LOG_INFO("{} took {:.1f} ms", operation_name, milliseconds);
    return milliseconds;
}

// Template kernel implementations
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

// Optimized dual scatter kernel - processes both PSV and NSV in a single pass
// Reduces memory bandwidth by reading SA indices only once
template<typename SA_t>
__global__ void dualScatterKernel(
    const SA_t* __restrict__ psv_values,  // PSV values (SA-order)
    const SA_t* __restrict__ nsv_values,  // NSV values (SA-order)
    const SA_t* __restrict__ indices,     // SA array chunk (text positions)
    SA_t* __restrict__ psv_output,        // PSV output array (text-order)
    SA_t* __restrict__ nsv_output,        // NSV output array (text-order)
    size_t chunk_size
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= chunk_size) return;

    size_t text_pos = indices[tid];       // Read SA index once
    psv_output[text_pos] = psv_values[tid];
    nsv_output[text_pos] = nsv_values[tid];
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
    LOG_INFO("\n=== Full GPU Memory Profile ===");
    LOG_INFO("GPU Total Memory: {:.2f} MB", total_mem / (1024.0 * 1024.0));
    LOG_INFO("GPU Free Memory (start): {:.2f} MB", free_mem_start / (1024.0 * 1024.0));
    LOG_INFO("Input SA already on GPU: {:.2f} MB", length * sizeof(SA_t) / (1024.0 * 1024.0));

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

    LOG_INFO("Allocated combined buffer: {:.2f} MB", total_size / (1024.0 * 1024.0));
    LOG_INFO("  - d_psv_sa_order: {:.2f} MB", length * sizeof(SA_t) / (1024.0 * 1024.0));
    LOG_INFO("  - d_nsv_sa_order: {:.2f} MB", length * sizeof(SA_t) / (1024.0 * 1024.0));
    LOG_INFO("  - d_temp (shared with block_mins): {:.2f} MB (first {} blocks = {:.2f} KB)",
             length * sizeof(SA_t) / (1024.0 * 1024.0), num_blocks, num_blocks * sizeof(SA_t) / 1024.0);

    size_t free_mem_after_alloc;
    cudaMemGetInfo(&free_mem_after_alloc, &total_mem);
    LOG_INFO("GPU Free Memory (after alloc): {:.2f} MB", free_mem_after_alloc / (1024.0 * 1024.0));
    LOG_INFO("Peak GPU Memory Used: {:.2f} MB", (free_mem_start - free_mem_after_alloc) / (1024.0 * 1024.0));
    LOG_INFO("================================\n");

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
    LOG_INFO("\nGPU Free Memory (after cleanup): {:.2f} MB\n", free_mem_end / (1024.0 * 1024.0));

    profiler.stop("Full GPU Processing (Single Work Array)");

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG_ERROR("CUDA error: {}", cudaGetErrorString(err));
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

    LOG_INFO("  PSV resolved {} positions (searched: {} blocks, skipped: {} blocks)",
             unfound_indices.size(), blocks_searched, blocks_skipped);
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

    LOG_INFO("  NSV resolved {} positions (searched: {} blocks, skipped: {} blocks)",
             unfound_indices.size(), blocks_searched, blocks_skipped);
}

template<typename SA_t>
void PipelinePSVNSVProcessor::processWithStreams(std::vector<SA_t>& sa_array, const uint8_t* data, size_t length,
                                                  const std::string& output_prefix, bool use_uint40) {
    // uint40 optimization: only applies to size_t mode
    const bool uint40_enabled = use_uint40 && std::is_same_v<SA_t, size_t>;

    // If uint40 enabled, compress SA first
    std::unique_ptr<uint40_vector> sa_compressed;
    if (uint40_enabled) {
        LOG_INFO("\n=== uint40 Optimization: Compressing SA ===");
        profiler.start();
        sa_compressed = std::make_unique<uint40_vector>(length);

        // Need to convert SA_t to size_t for pack_from
        std::vector<size_t> sa_size_t(sa_array.begin(), sa_array.end());
        sa_compressed->pack_from(sa_size_t);
        profiler.stop("SA compression to uint40");

        // Free original SA
        size_t sa_freed_mb = (length * sizeof(SA_t)) / (1024.0 * 1024.0);
        size_t sa_uint40_mb = (length * 5) / (1024.0 * 1024.0);
        LOG_INFO("  Freed size_t SA: {:.2f} MB", sa_freed_mb);
        LOG_INFO("  uint40 SA: {:.2f} MB", sa_uint40_mb);
        LOG_INFO("  Memory saved: {:.2f} MB ({:.1f}%%)", sa_freed_mb - sa_uint40_mb,
                 (1.0 - 5.0/sizeof(SA_t)) * 100);

        sa_array.clear();
        sa_array.shrink_to_fit();
    }

    const SA_t* sa_ptr = uint40_enabled ? nullptr : sa_array.data();

    try {
        LOG_INFO("\n=== Starting Optimized Stream Processing ===");
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

        LOG_INFO("\nConfiguration:");
        LOG_INFO("  Total chunks: {}", num_chunks);
        LOG_INFO("  Chunk size: {} MB", chunk_size / (1024 * 1024));

        // Allocate result arrays (use uint40 if enabled)
        std::unique_ptr<uint40_vector> psv_uint40, nsv_uint40;
        std::vector<SA_t> psv_results, nsv_results;

        if (uint40_enabled) {
            size_t cpu_alloc_size = 2 * length * 5;  // uint40 = 5 bytes
            LOG_INFO("\nCPU Memory Allocation (uint40):");
            LOG_INFO("  PSV + NSV results: {:.2f} MB", cpu_alloc_size / (1024.0 * 1024.0));
            psv_uint40 = std::make_unique<uint40_vector>(length);
            nsv_uint40 = std::make_unique<uint40_vector>(length);

            size_t saved = 2 * length * (sizeof(SA_t) - 5);
            LOG_INFO("  Memory saved vs size_t: {:.2f} MB", saved / (1024.0 * 1024.0));
        } else {
            size_t cpu_alloc_size = 2 * length * sizeof(SA_t);
            LOG_INFO("\nCPU Memory Allocation:");
            LOG_INFO("  PSV + NSV results: {:.2f} MB", cpu_alloc_size / (1024.0 * 1024.0));
            psv_results.resize(length, get_max_value<SA_t>());
            nsv_results.resize(length, get_max_value<SA_t>());
        }

        size_t sa_memory = uint40_enabled ? (length * 5) : (sa_array.size() * sizeof(SA_t));
        size_t input_memory = length * sizeof(uint8_t);
        size_t cpu_alloc_size = uint40_enabled ? (2 * length * 5) : (2 * length * sizeof(SA_t));
        double estimated_peak_mb = (cpu_alloc_size + sa_memory + input_memory) / (1024.0 * 1024.0);
        LOG_INFO("  Input data (resident in main): {:.2f} MB", input_memory / (1024.0 * 1024.0));
        LOG_INFO("  SA (CPU): {:.2f} MB", sa_memory / (1024.0 * 1024.0));
        LOG_INFO("  Estimated CPU peak before metadata: {:.2f} MB (data + SA + PSV + NSV)", estimated_peak_mb);

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
        LOG_INFO("  Block mins metadata: {:.4f} MB", block_mins_size / (1024.0 * 1024.0));
        std::vector<SA_t> global_block_mins(total_blocks);
        LOG_INFO("  Estimated CPU peak with metadata: {:.2f} MB (actual peak may vary with unfound indices)",
                 estimated_peak_mb + block_mins_size / (1024.0 * 1024.0));

        // ======== PHASE 1: GPU Chunk Processing + Metadata Collection ========
        LOG_INFO("\n=== Phase 1: GPU Chunk Processing ===");
        size_t free_before, total_gpu;
        cudaMemGetInfo(&free_before, &total_gpu);
        LOG_INFO("GPU Memory before Phase 1: {:.2f} MB free / {:.2f} MB total",
                 free_before / (1024.0 * 1024.0), total_gpu / (1024.0 * 1024.0));

        profiler.start();
        {
            cudaStream_t compute_stream;
            cudaStreamCreate(&compute_stream);

            const size_t shared_mem_size = block_size * sizeof(SA_t);

            // Allocate GPU buffers for one chunk
            size_t gpu_chunk_alloc = 3 * chunk_size * sizeof(SA_t);
            LOG_INFO("GPU per-chunk allocation: {:.2f} MB (input + psv + nsv)", gpu_chunk_alloc / (1024.0 * 1024.0));

            SA_t *d_input, *d_psv_output, *d_nsv_output, *d_block_mins;
            cudaMalloc(&d_input, chunk_size * sizeof(SA_t));
            cudaMalloc(&d_psv_output, chunk_size * sizeof(SA_t));
            cudaMalloc(&d_nsv_output, chunk_size * sizeof(SA_t));

            // Temp buffer for uint40 unpacking (reused across chunks)
            std::vector<uint64_t> sa_chunk_buffer;
            if (uint40_enabled) {
                sa_chunk_buffer.reserve(chunk_size);
            }

            for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
                size_t offset = chunk_idx * chunk_size;
                size_t current_chunk_size = std::min(chunk_size, length - offset);
                const int num_blocks = (current_chunk_size + block_size - 1) / block_size;

                float progress = ((chunk_idx + 1) * 100.0f) / num_chunks;
                fmt::print("\rPhase 1 - GPU Processing: {:.1f}% [{}/{}]", progress, chunk_idx + 1, num_chunks);
                std::fflush(stdout);

                cudaMalloc(&d_block_mins, num_blocks * sizeof(SA_t));

                // Upload chunk to GPU (with uint40 conversion if needed)
                if (uint40_enabled) {
                    // Unpack uint40 -> size_t (OpenMP parallel)
                    sa_chunk_buffer.clear();
                    sa_compressed->unpack_range(offset, current_chunk_size, sa_chunk_buffer);

                    cudaMemcpyAsync(d_input, sa_chunk_buffer.data(),
                                  current_chunk_size * sizeof(SA_t),
                                  cudaMemcpyHostToDevice,
                                  compute_stream);
                } else {
                    cudaMemcpyAsync(d_input, sa_ptr + offset,
                                  current_chunk_size * sizeof(SA_t),
                                  cudaMemcpyHostToDevice,
                                  compute_stream);
                }

                // Step 1: Compute PSV/NSV within blocks
                computePSVNSVKernel<<<num_blocks, block_size, shared_mem_size, compute_stream>>>(
                    d_input, d_psv_output, d_nsv_output, d_block_mins, current_chunk_size
                );

                // Step 2: Process cross-block boundaries within this chunk (GPU merge)
                processPSVNSVBoundariesKernel<<<num_blocks, block_size, 0, compute_stream>>>(
                    d_input, d_psv_output, d_nsv_output, d_block_mins,
                    current_chunk_size, block_size
                );

                // Download results and block_mins (with uint40 conversion if needed)
                std::vector<uint64_t> psv_temp, nsv_temp;
                if (uint40_enabled) {
                    // Download to temp buffer, then pack to uint40
                    psv_temp.resize(current_chunk_size);
                    nsv_temp.resize(current_chunk_size);

                    cudaMemcpyAsync(psv_temp.data(), d_psv_output,
                                  current_chunk_size * sizeof(SA_t),
                                  cudaMemcpyDeviceToHost, compute_stream);
                    cudaMemcpyAsync(nsv_temp.data(), d_nsv_output,
                                  current_chunk_size * sizeof(SA_t),
                                  cudaMemcpyDeviceToHost, compute_stream);
                } else {
                    cudaMemcpyAsync(&psv_results[offset], d_psv_output,
                                  current_chunk_size * sizeof(SA_t),
                                  cudaMemcpyDeviceToHost, compute_stream);
                    cudaMemcpyAsync(&nsv_results[offset], d_nsv_output,
                                  current_chunk_size * sizeof(SA_t),
                                  cudaMemcpyDeviceToHost, compute_stream);
                }

                // Save block_mins for this chunk to global array
                size_t block_offset = (offset / block_size);
                cudaMemcpyAsync(&global_block_mins[block_offset], d_block_mins,
                              num_blocks * sizeof(SA_t),
                              cudaMemcpyDeviceToHost, compute_stream);

                cudaStreamSynchronize(compute_stream);
                cudaFree(d_block_mins);

                // Pack results to uint40 if enabled (OpenMP parallel)
                if (uint40_enabled) {
                    psv_uint40->pack_range(offset, psv_temp, 0, current_chunk_size);
                    nsv_uint40->pack_range(offset, nsv_temp, 0, current_chunk_size);
                }

                // Collect metadata: unfound positions in this chunk
                // After GPU boundary merge, these should only be inter-chunk boundaries
                const uint64_t MAX_VAL = uint40_enabled ? SIZE_MAX : get_max_value<SA_t>();
                for (size_t i = offset; i < offset + current_chunk_size; ++i) {
                    uint64_t psv_val = uint40_enabled ? psv_temp[i - offset] : psv_results[i];
                    if (psv_val == MAX_VAL) {
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

                    uint64_t nsv_val = uint40_enabled ? nsv_temp[i - offset] : nsv_results[i];
                    if (nsv_val == MAX_VAL) {
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

            fmt::print("\n");
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
        LOG_INFO("GPU Memory after Phase 1: {:.1f} MB free", free_after_phase1 / (1024.0 * 1024.0));

        LOG_INFO("\nMetadata collected (after GPU intra-chunk merge):");
        LOG_INFO("  PSV unfound (inter-chunk boundaries): {} positions ({:.1f}%)",
                 global_psv_unfound.size(), global_psv_unfound.size() * 100.0 / length);
        LOG_INFO("  NSV unfound (inter-chunk boundaries): {} positions ({:.1f}%)",
                 global_nsv_unfound.size(), global_nsv_unfound.size() * 100.0 / length);

        if (num_chunks > 1) {
            LOG_INFO("  Note: GPU handled intra-chunk boundaries, CPU will handle {} inter-chunk boundaries",
                     num_chunks - 1);
        }

        // ======== PHASE 2: CPU Targeted Search with Block Min Pruning ========
        LOG_INFO("\n=== Phase 2: CPU Targeted Search ===");
        profiler.start();

        if (uint40_enabled) {
            // uint40 path: direct access via get/set (slower but memory-efficient)
            if (!global_psv_unfound.empty()) {
                const SA_t MAX_VAL = get_max_value<SA_t>();
                size_t blocks_searched = 0, blocks_skipped = 0;

                #pragma omp parallel for reduction(+:blocks_searched,blocks_skipped)
                for (size_t idx = 0; idx < global_psv_unfound.size(); ++idx) {
                    size_t pos = global_psv_unfound[idx];
                    uint64_t current = sa_compressed->get(pos);
                    size_t current_block = pos / block_size;

                    for (int block = static_cast<int>(current_block) - 1; block >= 0; --block) {
                        if (global_block_mins[block] >= current) {
                            blocks_skipped++;
                            continue;
                        }

                        size_t block_start = block * block_size;
                        size_t block_end = std::min(static_cast<size_t>((block + 1) * block_size), length);

                        blocks_searched++;
                        for (size_t i = std::min(pos, block_end) - 1; i >= block_start; --i) {
                            if (sa_compressed->get(i) < current) {
                                psv_uint40->set(pos, sa_compressed->get(i));
                                goto found_psv_uint40;
                            }
                            if (i == 0) break;
                        }
                    }
                    psv_uint40->set(pos, MAX_VAL);
                    found_psv_uint40:;
                }
                LOG_INFO("  PSV resolved {} positions (searched: {} blocks, skipped: {} blocks)",
                         global_psv_unfound.size(), blocks_searched, blocks_skipped);
            }

            if (!global_nsv_unfound.empty()) {
                const SA_t MAX_VAL = get_max_value<SA_t>();
                size_t total_blocks = (length + block_size - 1) / block_size;
                size_t blocks_searched = 0, blocks_skipped = 0;

                #pragma omp parallel for reduction(+:blocks_searched,blocks_skipped)
                for (size_t idx = 0; idx < global_nsv_unfound.size(); ++idx) {
                    size_t pos = global_nsv_unfound[idx];
                    uint64_t current = sa_compressed->get(pos);
                    size_t current_block = pos / block_size;

                    for (size_t block = current_block + 1; block < total_blocks; ++block) {
                        if (global_block_mins[block] >= current) {
                            blocks_skipped++;
                            continue;
                        }

                        size_t block_start = block * block_size;
                        size_t block_end = std::min(static_cast<size_t>((block + 1) * block_size), length);

                        blocks_searched++;
                        for (size_t i = std::max(pos + 1, block_start); i < block_end; ++i) {
                            if (sa_compressed->get(i) < current) {
                                nsv_uint40->set(pos, sa_compressed->get(i));
                                goto found_nsv_uint40;
                            }
                        }
                    }
                    nsv_uint40->set(pos, MAX_VAL);
                    found_nsv_uint40:;
                }
                LOG_INFO("  NSV resolved {} positions (searched: {} blocks, skipped: {} blocks)",
                         global_nsv_unfound.size(), blocks_searched, blocks_skipped);
            }
        } else {
            // Standard path: direct array access
            if (!global_psv_unfound.empty()) {
                resolvePSVWithPruning(sa_ptr, psv_results.data(), global_psv_unfound,
                                     global_block_mins, length, block_size);
            }

            if (!global_nsv_unfound.empty()) {
                resolveNSVWithPruning(sa_ptr, nsv_results.data(), global_nsv_unfound,
                                     global_block_mins, length, block_size);
            }
        }

        // free metadata vectors
        global_psv_unfound.clear();
        global_psv_unfound.shrink_to_fit();
        global_nsv_unfound.clear();
        global_nsv_unfound.shrink_to_fit();

        profiler.stop("Phase 2: CPU Targeted Search");

        // ======== PHASE 3: Text Order Conversion ========
        LOG_INFO("\n=== Phase 3: SA-order to Text-order Conversion ===");

        if (uint40_enabled) {
            // uint40 path: need temp buffers for GPU conversion
            // This is the most memory-intensive step, but unavoidable
            LOG_INFO("  Unpacking uint40 arrays for GPU text-order conversion...");
            profiler.start();

            std::vector<uint64_t> sa_full, psv_full, nsv_full;
            sa_compressed->unpack_all(sa_full);
            psv_uint40->unpack_all(psv_full);
            nsv_uint40->unpack_all(nsv_full);

            profiler.stop("uint40 unpacking");

            // Convert to text order using GPU
            convertToTextOrderGPUStreaming(sa_full.data(), psv_full.data(), nsv_full.data(), length);

            // Move results back (psv_full and nsv_full are already in text order now)
            // Need to cast uint64_t -> SA_t
            psv_results.resize(length);
            nsv_results.resize(length);
            #pragma omp parallel for
            for (size_t i = 0; i < length; ++i) {
                psv_results[i] = static_cast<SA_t>(psv_full[i]);
                nsv_results[i] = static_cast<SA_t>(nsv_full[i]);
            }

            // Free uint40 and SA
            size_t sa_freed = (length * 5) / (1024.0 * 1024.0);
            sa_compressed.reset();
            psv_uint40.reset();
            nsv_uint40.reset();
            sa_full.clear();
            sa_full.shrink_to_fit();
            LOG_INFO("  Freed uint40 SA/PSV/NSV: {:.2f} MB", sa_freed * 3);

        } else {
            // Standard path
            convertToTextOrderGPUStreaming(sa_ptr, psv_results.data(), nsv_results.data(), length);

            size_t sa_memory_mb = (sa_array.size() * sizeof(SA_t)) / (1024.0 * 1024.0);
            LOG_INFO("\nReleasing SA (freeing {} MB)", sa_memory_mb);
            sa_array.clear();
            sa_array.shrink_to_fit();
        }

        // ======== PHASE 4: LZ77 Factorization ========
        LOG_INFO("\n=== Phase 4: LZ77 Factorization ===");
        profiler.start();
        std::string lz_output = output_prefix + "_lz77.bin";
        ComputeLZ77(data, psv_results.data(), nsv_results.data(), length - 1, lz_output);
        profiler.stop("LZ77 Processing");

        LOG_INFO("\n=== Stream Processing Complete ===");

    } catch (const std::exception& e) {
        LOG_ERROR("Error in stream processing: {}", e.what());
        throw;
    }
}

template<typename SA_t>
void PipelinePSVNSVProcessor::convertToTextOrderGPUStreaming(
    const SA_t* sa_array,
    SA_t* psv,
    SA_t* nsv,
    size_t length) {

    profiler.start();

    // Calculate available GPU memory and optimal chunk size
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    // Memory requirements per chunk:
    // - d_sa_chunk: 1x chunk_size (input SA chunk)
    // - d_psv_in: 1x chunk_size (input PSV chunk)
    // - d_nsv_in: 1x chunk_size (input NSV chunk)
    // - d_psv_out: 1x length (full output PSV, reused)
    // - d_nsv_out: 1x length (full output NSV, reused)
    // Total: (3 * chunk_size + 2 * length) * sizeof(SA_t)

    size_t usable = free_mem * 0.9;  // Leave 15% safety margin
    size_t output_size = 2 * length * sizeof(SA_t);  // d_psv_out + d_nsv_out

    if (usable <= output_size) {
        throw std::runtime_error(
            "Insufficient GPU memory for streaming text-order conversion. "
            "Free: " + std::to_string(free_mem / (1024.0 * 1024.0)) + " MB, "
            "Required: " + std::to_string(output_size / (1024.0 * 1024.0)) + " MB"
        );
    }

    size_t remaining = usable - output_size;
    size_t chunk_size = remaining / (3 * sizeof(SA_t));
    chunk_size = std::min(chunk_size, length);

    // Align to warp size for efficiency
    chunk_size = (chunk_size / 32) * 32;
    if (chunk_size == 0) chunk_size = 32;

    size_t num_chunks = (length + chunk_size - 1) / chunk_size;

    LOG_INFO("\n=== GPU Streaming Text-Order Conversion ===");
    LOG_INFO("  GPU Free Memory: {:.1f} MB", free_mem / (1024.0 * 1024.0));
    LOG_INFO("  Output buffers (full): {:.1f} MB", output_size / (1024.0 * 1024.0));
    LOG_INFO("  Chunk size: {} elements ({:.1f} MB)",
             chunk_size, (chunk_size * sizeof(SA_t)) / (1024.0 * 1024.0));
    LOG_INFO("  Total chunks: {}", num_chunks);

    // Allocate GPU buffers
    SA_t *d_sa_chunk, *d_psv_in, *d_psv_out, *d_nsv_in, *d_nsv_out;
    cudaMalloc(&d_sa_chunk, chunk_size * sizeof(SA_t));
    cudaMalloc(&d_psv_in, chunk_size * sizeof(SA_t));
    cudaMalloc(&d_psv_out, length * sizeof(SA_t));
    cudaMalloc(&d_nsv_in, chunk_size * sizeof(SA_t));
    cudaMalloc(&d_nsv_out, length * sizeof(SA_t));

    // Initialize output buffers to MAX_VAL (shouldn't be needed, but for safety)
    const SA_t MAX_VAL = get_max_value<SA_t>();
    thrust::device_ptr<SA_t> psv_out_ptr = thrust::device_pointer_cast(d_psv_out);
    thrust::device_ptr<SA_t> nsv_out_ptr = thrust::device_pointer_cast(d_nsv_out);
    thrust::fill(psv_out_ptr, psv_out_ptr + length, MAX_VAL);
    thrust::fill(nsv_out_ptr, nsv_out_ptr + length, MAX_VAL);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // ======== Optimized: Process PSV and NSV together ========
    profiler.start();
    LOG_INFO("\nProcessing PSV+NSV (dual scatter):");
    for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        size_t offset = chunk_idx * chunk_size;
        size_t current_chunk = std::min(chunk_size, length - offset);

        float progress = ((chunk_idx + 1) * 100.0f) / num_chunks;
        fmt::print("\r  Dual scatter: {:.1f}% [{}/{}]", progress, chunk_idx + 1, num_chunks);
        std::fflush(stdout);

        // Upload SA chunk, PSV chunk, and NSV chunk
        cudaMemcpyAsync(d_sa_chunk, sa_array + offset, current_chunk * sizeof(SA_t),
                       cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_psv_in, psv + offset, current_chunk * sizeof(SA_t),
                       cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_nsv_in, nsv + offset, current_chunk * sizeof(SA_t),
                       cudaMemcpyHostToDevice, stream);

        cudaStreamSynchronize(stream);

        // Dual scatter: process both PSV and NSV in one kernel
        int blocks = (current_chunk + 255) / 256;
        dualScatterKernel<<<blocks, 256, 0, stream>>>(
            d_psv_in, d_nsv_in, d_sa_chunk, d_psv_out, d_nsv_out, current_chunk
        );
    }
    cudaStreamSynchronize(stream);
    fmt::print("\n");
    profiler.stop("  Dual scatter (GPU)");

    // Download results
    profiler.start();
    cudaMemcpy(psv, d_psv_out, length * sizeof(SA_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(nsv, d_nsv_out, length * sizeof(SA_t), cudaMemcpyDeviceToHost);
    profiler.stop("  PSV+NSV download");

    // Cleanup
    cudaStreamDestroy(stream);
    cudaFree(d_sa_chunk);
    cudaFree(d_psv_in);
    cudaFree(d_psv_out);
    cudaFree(d_nsv_in);
    cudaFree(d_nsv_out);

    profiler.stop("Total GPU streaming text-order conversion");
}

// Explicit template instantiations
template void PipelinePSVNSVProcessor::processFullGPUWithGPUSA<uint32_t>(uint32_t*, const uint8_t*, size_t, const std::string&);
template void PipelinePSVNSVProcessor::processFullGPUWithGPUSA<size_t>(size_t*, const uint8_t*, size_t, const std::string&);

template void PipelinePSVNSVProcessor::resolvePSVWithPruning<uint32_t>(const uint32_t*, uint32_t*, const std::vector<size_t>&, const std::vector<uint32_t>&, size_t, size_t);
template void PipelinePSVNSVProcessor::resolvePSVWithPruning<size_t>(const size_t*, size_t*, const std::vector<size_t>&, const std::vector<size_t>&, size_t, size_t);

template void PipelinePSVNSVProcessor::resolveNSVWithPruning<uint32_t>(const uint32_t*, uint32_t*, const std::vector<size_t>&, const std::vector<uint32_t>&, size_t, size_t);
template void PipelinePSVNSVProcessor::resolveNSVWithPruning<size_t>(const size_t*, size_t*, const std::vector<size_t>&, const std::vector<size_t>&, size_t, size_t);

template void PipelinePSVNSVProcessor::processWithStreams<uint32_t>(std::vector<uint32_t>&, const uint8_t*, size_t, const std::string&, bool);
template void PipelinePSVNSVProcessor::processWithStreams<size_t>(std::vector<size_t>&, const uint8_t*, size_t, const std::string&, bool);

template std::pair<std::pair<size_t, size_t>, size_t> PipelinePSVNSVProcessor::LZFactor<uint32_t>(const uint8_t*, size_t, uint32_t, uint32_t, size_t);
template std::pair<std::pair<size_t, size_t>, size_t> PipelinePSVNSVProcessor::LZFactor<size_t>(const uint8_t*, size_t, size_t, size_t, size_t);

template void PipelinePSVNSVProcessor::ComputeLZ77<uint32_t>(const uint8_t*, uint32_t*, uint32_t*, size_t, std::string);
template void PipelinePSVNSVProcessor::ComputeLZ77<size_t>(const uint8_t*, size_t*, size_t*, size_t, std::string);

template void PipelinePSVNSVProcessor::convertToTextOrderGPUStreaming<uint32_t>(const uint32_t*, uint32_t*, uint32_t*, size_t);
template void PipelinePSVNSVProcessor::convertToTextOrderGPUStreaming<size_t>(const size_t*, size_t*, size_t*, size_t);

// Explicit kernel instantiations
template __global__ void computePSVNSVKernel<uint32_t>(const uint32_t*, uint32_t*, uint32_t*, uint32_t*, const size_t);
template __global__ void computePSVNSVKernel<size_t>(const size_t*, size_t*, size_t*, size_t*, const size_t);

template __global__ void processPSVNSVBoundariesKernel<uint32_t>(const uint32_t*, uint32_t*, uint32_t*, const uint32_t*, const size_t, const size_t);
template __global__ void processPSVNSVBoundariesKernel<size_t>(const size_t*, size_t*, size_t*, const size_t*, const size_t, const size_t);

template __global__ void dualScatterKernel<uint32_t>(const uint32_t*, const uint32_t*, const uint32_t*, uint32_t*, uint32_t*, size_t);
template __global__ void dualScatterKernel<size_t>(const size_t*, const size_t*, const size_t*, size_t*, size_t*, size_t);
