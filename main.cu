#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <string>
#include <stack>
#include <algorithm>
#include <numeric>
#include "libcubwt.cuh"
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/copy.h>
#include <cuda_runtime.h>
#include "processor.cuh"
#include "timer.cuh"
#include "processor.cuh"

int main(int argc, char **argv) {
    // CUDA Event Creation for timing
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /*  Read Input Files and construct Data Vector */

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
        return 1;
    }

    std::string input_file = argv[1];
    std::string output_file = argv[2];
    std::string sa_output = input_file + ".sa.txt";
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
    uint32_t *SA  = new uint32_t[length];

    // allocate device storage
    void *device_storage = nullptr;
    int result = libcubwt_allocate_device_storage(&device_storage, length);
    if (result != LIBCUBWT_NO_ERROR) {
        printf("Allocation error: %d\n", result);
        return 1;
    }

    // Record start event
    cudaEventRecord(start);

    // generate suffix array
    result = libcubwt_sa(device_storage, data.data(), SA, length);
    if (result != LIBCUBWT_NO_ERROR) {
        printf("Suffix array error: %d\n", result);
        return 1;
    }


    // Record stop event and calculate elapsed time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time for computing SA: " << milliseconds << " ms" << std::endl;

    PipelinePSVNSVProcessor processor(6, 256, 16384, 2);
    processor.process(SA, length, output_file);

    // Free the allocated memory on the GPU to avoid memory leaks

    delete[] SA;

    return 0;
}



