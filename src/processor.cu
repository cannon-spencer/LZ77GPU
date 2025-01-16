//
// Created by yu hong on 11/17/24.
//

#include "processor.cuh"

PipelinePSVNSVProcessor::PipelinePSVNSVProcessor(int streams, size_t block_size, size_t shared_mem_size,
                                                 int batch_div)
                                                 : numStreams(streams),
                                                   blockSize(block_size),
                                                   sharedMemSize(shared_mem_size),
                                                   batchDivisor(batch_div){
    initializeStreams();
}

PipelinePSVNSVProcessor::~PipelinePSVNSVProcessor() {
    if (streams) {
        for (int i = 0; i < numStreams; ++i) {
            cudaStreamDestroy(streams[i]);
        }
        delete[] streams;
    }
}

void PipelinePSVNSVProcessor::initializeStreams() {
    std::cout<<numStreams<<"\n";
    streams = new cudaStream_t[numStreams];
    for (int i = 0; i < numStreams; ++i) {
        cudaStreamCreate(&streams[i]);
    }
}

void PipelinePSVNSVProcessor::calculateOptimalBatchSize(size_t length, size_t &batchSize) {
    // Get GPU properties
    std::cout<<"hereeeee \n";
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);  // Assume using device 0

    // Get available GPU memory
    size_t free_memory, total_memory;
    cudaMemGetInfo(&free_memory, &total_memory);

    // Base calculations
    size_t optimal_blocks_per_sm = 2;  // Base value for optimal blocks per SM
    size_t total_sms = prop.multiProcessorCount;
    size_t max_threads_per_sm = prop.maxThreadsPerMultiProcessor;

    // Calculate theoretical maximum batch size
    size_t optimal_total_blocks = optimal_blocks_per_sm * total_sms;
    size_t theoretical_batch_size = blockSize * optimal_total_blocks;

    // Calculate memory requirements for streams
    size_t element_size = sizeof(Element);
    size_t memory_per_stream = 3 * element_size;  // For input, psv_output, nsv_output
    size_t total_required_memory = memory_per_stream * numStreams;

    // Calculate memory-limited batch size using all available memory
    size_t memory_limited_batch_size = free_memory / total_required_memory;

    // Adjust based on input size
    size_t input_limited_batch_size = length / 4;  // Ensure at least 4 batches for parallelism
    if (input_limited_batch_size < theoretical_batch_size) {
        theoretical_batch_size = input_limited_batch_size;
    }

    // Take minimum of all constraints
    batchSize = std::min({
        theoretical_batch_size,
        memory_limited_batch_size,
        length  // Don't exceed total length
    });

    // Ensure batch size is multiple of blockSize
    batchSize = (batchSize / blockSize) * blockSize;

    // Ensure minimum batch size
    size_t min_batch_size = blockSize * total_sms;
    batchSize = std::max(batchSize, min_batch_size);

    // Print calculation details
    std::cout << "\nBatch Size Calculation Details:\n"
              << "- GPU: " << prop.name << "\n"
              << "- Number of SMs: " << total_sms << "\n"
              << "- Available Memory: " << free_memory / (1024*1024) << " MB\n"
              << "- Maximum Theoretical Batch: " << theoretical_batch_size << "\n"
              << "- Memory Constrained Batch: " << memory_limited_batch_size << "\n"
              << "- Input Size Limited Batch: " << input_limited_batch_size << "\n"
              << "- Final Batch Size: " << batchSize << "\n"
              << "- Expected Number of Batches: " << (length + batchSize - 1) / batchSize << "\n\n";
}

void PipelinePSVNSVProcessor::processBlockBoundary(std::vector<Element> &psv_results,
                                                  std::vector<Element> &nsv_results,
                                                  const uint32_t* input,
                                                  size_t totalSize) {
    // Calculate total number of blocks
    size_t totalBlocks = (totalSize + blockSize - 1) / blockSize;

    std::cout << "Processing boundaries between blocks\n";
    std::cout << "Total blocks: " << totalBlocks << "\n";

    // Process PSV boundaries between blocks
    for (size_t block = 1; block < totalBlocks; ++block) {
        size_t blockStart = block * blockSize;
        size_t blockEnd = std::min(blockStart + blockSize, totalSize);

        // For each element in current block
        for (size_t i = blockStart; i < blockEnd; ++i) {
            // Only process elements whose PSV is INT_MAX
            if (psv_results[i].value == INT32_MAX) {
                // Check elements in previous blocks
                for (size_t j = blockStart - 1; j != (size_t)-1; --j) {
                    if (input[j] < input[i]) {
                        psv_results[i] = Element(input[j], j);
                        break;  // Found the closest previous smaller value
                    }
                }
            }
        }
    }

    // Process NSV boundaries between blocks
    for (size_t block = 0; block < totalBlocks - 1; ++block) {
        size_t blockStart = block * blockSize;
        size_t blockEnd = std::min((block + 1) * blockSize, totalSize);

        // For each element in current block
        for (size_t i = blockStart; i < blockEnd; ++i) {
            // Only process elements whose NSV is INT_MAX
            if (nsv_results[i].value == INT32_MAX) {
                // Check elements in following blocks
                for (size_t j = blockEnd; j < totalSize; ++j) {
                    if (input[j] < input[i]) {
                        nsv_results[i] = Element(input[j], j);
                        break;  // Found the closest next smaller value
                    }
                }
            }
        }
    }
}

void PipelinePSVNSVProcessor::processTextOrder(const uint32_t* input,
                                               const std::vector <Element> &psv_results,
                                               const std::vector <Element> &nsv_results,
                                               const std::string &output_prefix,
                                               size_t length) {
    Timer timer("Text Order Processing");

    size_t n = length;

    size_t required_memory = sizeof(uint32_t) * n;

    std::vector<uint32_t> psv_text_order(n, UINT32_MAX);
    std::vector<int64_t> nsv_text_order(n, UINT32_MAX);

    if(required_memory > DEFAULT_GPU_MEMORY){
        std::cout << "Using CPU for text order (required memory: "
                  << required_memory / (1024*1024*1024) << "GB)\n";

#pragma omp parallel for
        for(size_t i = 0; i < n; i++) {
            psv_text_order[input[i]] = psv_results[i].value;
            nsv_text_order[input[i]] = nsv_results[i].value;
        }
    }else{
        std::cout << "Using GPU for text order (required memory: "
                  << required_memory / (1024*1024*1024) << "GB)\n";

//        std::vector<uint32_t> input_vec(input, input + n);
        thrust::device_vector<uint32_t> d_input(input, input + n);
        thrust::device_vector<Element> d_psv_results = psv_results;
        thrust::device_vector<Element> d_nsv_results = nsv_results;

        thrust::device_vector<uint32_t> d_psv_values(n);
        thrust::device_vector<uint32_t> d_psv_text_order(n, UINT32_MAX);
        thrust::transform(d_psv_results.begin(), d_psv_results.end(),
                          d_psv_values.begin(),
                          [] __device__ (const Element& e){
                              return e.value;
                          });
        thrust::scatter(d_psv_values.begin(), d_psv_values.end(),
                        d_input.begin(), d_psv_text_order.begin());

        thrust::device_vector<int64_t> d_nsv_values(n);
        thrust::device_vector<int64_t> d_nsv_text_order(n, INT64_MAX);
        thrust::transform(d_nsv_results.begin(), d_nsv_results.end(),
                          d_nsv_values.begin(),
                          [] __device__ (const Element& e) {
                              return e.value;
                          });
        thrust::scatter(d_nsv_values.begin(), d_nsv_values.end(),
                        d_input.begin(), d_nsv_text_order.begin());

        // copy back to CPU
        thrust::copy(d_psv_text_order.begin(), d_psv_text_order.end(),
                     psv_text_order.begin());
        thrust::copy(d_nsv_text_order.begin(), d_nsv_text_order.end(),
                     nsv_text_order.begin());
    }

//    std::cout << "\nText Order Results (first 10 elements or less):\n";
//    size_t print_count = std::min(n, size_t(30));
//
//    std::cout << "PSV Text Order:\n";
//    for (size_t i = 0; i < print_count; ++i) {
//        std::cout << "Index " << i << ": " << psv_text_order[i] << "\n";
//    }
    std::ofstream psv_file(output_prefix + "_psv.bin", std::ios::binary);
    std::ofstream nsv_file(output_prefix + "_nsv.bin", std::ios::binary);

    psv_file.write(reinterpret_cast<const char*>(psv_text_order.data()),
                   length * sizeof(uint32_t));
    nsv_file.write(reinterpret_cast<const char*>(nsv_text_order.data()),
                   length * sizeof(uint32_t));

    psv_file.close();
    nsv_file.close();
}


void PipelinePSVNSVProcessor::process(const uint32_t *sa_array, size_t length, const std::string &output_prefix) {
    Timer total_timer("Total Processing");
    std::cout<<"process \n";
    // Compute the Batch size
    size_t batchSize;
    calculateOptimalBatchSize(length, batchSize);


    std::cout << "Processing with:\n"
              << "- Total elements: " << length << "\n"
              << "- Block size: " << blockSize << "\n"
              << "- Batch size: " << batchSize << "\n";

    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    size_t numBatches = (length + batchSize - 1) / batchSize;
    std::cout << "- Number of batches: " << numBatches << "\n\n";

    Element *h_elements, *h_psv_results, *h_nsv_results;
    cudaHostAlloc(&h_elements, batchSize * sizeof(Element), cudaHostAllocDefault);
    cudaHostAlloc(&h_psv_results, batchSize * sizeof(Element), cudaHostAllocDefault);
    cudaHostAlloc(&h_nsv_results, batchSize * sizeof(Element), cudaHostAllocDefault);

    Element *d_input[DEFAULT_NUM_STREAMS], *d_psv_output[DEFAULT_NUM_STREAMS], *d_nsv_output[DEFAULT_NUM_STREAMS];
    for (int i = 0; i < numStreams; ++i) {
        cudaMalloc(&d_input[i], batchSize * sizeof(Element));
        cudaMalloc(&d_psv_output[i], batchSize * sizeof(Element));
        cudaMalloc(&d_nsv_output[i], batchSize * sizeof(Element));
    }

    std::vector <Element> psv_final_results(length);
    std::vector <Element> nsv_final_results(length);

    for (size_t batch = 0; batch < numBatches; ++batch) {
        int streamIdx = batch % numStreams;
        size_t batchStart = batch * batchSize;
        size_t currentBatchSize = std::min(batchSize, length - batchStart);

        for (size_t i = 0; i < currentBatchSize; ++i) {
            h_elements[i] = Element(sa_array[batchStart + i], batchStart + i);
        }

        cudaMemcpyAsync(d_input[streamIdx], h_elements,
                        currentBatchSize * sizeof(Element),
                        cudaMemcpyHostToDevice, streams[streamIdx]);

        int numBlocks = (currentBatchSize + blockSize - 1) / blockSize;

        blockPSVNSVKernel<<<numBlocks, blockSize, 0, streams[streamIdx]>>>(
                d_input[streamIdx],
                d_psv_output[streamIdx],
                d_nsv_output[streamIdx],
                blockSize,
                currentBatchSize
        );

        cudaMemcpyAsync(h_psv_results,
                        d_psv_output[streamIdx],
                        currentBatchSize * sizeof(Element),
                        cudaMemcpyDeviceToHost, streams[streamIdx]);
        cudaMemcpyAsync(h_nsv_results,
                        d_nsv_output[streamIdx],
                        currentBatchSize * sizeof(Element),
                        cudaMemcpyDeviceToHost, streams[streamIdx]);

        cudaStreamSynchronize(streams[streamIdx]);
        // copy the results to the final arrays
        std::memcpy(&psv_final_results[batchStart], h_psv_results,
                    currentBatchSize * sizeof(Element));
        std::memcpy(&nsv_final_results[batchStart], h_nsv_results,
                    currentBatchSize * sizeof(Element));

    }

    // merge the batch boundary
    processBlockBoundary(psv_final_results, nsv_final_results, sa_array, length);

    // Stop timing PSV/NSV computation
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "PSV/NSV Computation Time: " << milliseconds << " ms" << std::endl;

    cudaFreeHost(h_elements);
    cudaFreeHost(h_psv_results);
    cudaFreeHost(h_nsv_results);

    for (int i = 0; i < numStreams; ++i) {
        cudaFree(d_input[i]);
        cudaFree(d_psv_output[i]);
        cudaFree(d_nsv_output[i]);
    }
//    std::cout << "PSV NSV lex computed"<< std::endl;
//
//
//    std::cout << "PSV values:\n";
//    for (size_t i = 0; i < psv_final_results.size(); ++i) {
//        std::cout << i <<" "<< psv_final_results[i].value <<"\n";
//    }

    processTextOrder(sa_array, psv_final_results, nsv_final_results, output_prefix, length);

}