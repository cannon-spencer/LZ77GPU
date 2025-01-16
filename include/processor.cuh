//
// Created by yu hong on 11/17/24.
//

#ifndef LIBCUBWT_PROCESSOR_CUH
#define LIBCUBWT_PROCESSOR_CUH

#include "kernel.cuh"
#include "data_types.cuh"
#include "timer.cuh"
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <omp.h>

class PipelinePSVNSVProcessor {
private:
    int numStreams;
    size_t blockSize;
    size_t sharedMemSize;
    int batchDivisor;
    cudaStream_t* streams;

    void processBlockBoundary(std::vector<Element> &psv_results,
                                                  std::vector<Element> &nsv_results,
                                                  const uint32_t* input,
                                                  size_t totalSize);

    void initializeStreams();

    void calculateOptimalBatchSize(size_t length, size_t &batchSize);

public:
    PipelinePSVNSVProcessor(int streams = DEFAULT_NUM_STREAMS,
                            size_t block_size = DEFAULT_BLOCK_SIZE,
                            size_t shared_mem_size = DEFAULT_SHARED_MEMORY_SIZE,
                            int batch_div = DEFAULT_BATCH_DIV);

    ~PipelinePSVNSVProcessor();

    void process(const uint32_t* sa_array, size_t length, const std::string& output_prefix);

    void processTextOrder(const uint32_t* input,
                          const std::vector <Element> &psv_results,
                          const std::vector <Element> &nsv_results,
                          const std::string &output_prefix,
                          size_t length);


};


#endif //LIBCUBWT_PROCESSOR_CUH
