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

__global__ void computeBlockPSV(int32_t *d_A, int32_t *d_psv, size_t n) {
    // Calculate global thread index
    // threadIdx.x: thread's index within the block
    // blockIdx.x: block's index within the grid of blocks
    // blockDim.x: number of threads in each block (block dimension)
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    // Check if the global thread index exceeds the bounds of the array
    if (idx >= n) return;

    // Initialize the PSV for this element to -1
    d_psv[idx] = -1;

    // Iterate through previous elements to find the previous smaller value
    for (int j = idx - 1; j >= 0; --j) {
        // If a smaller element is found, store it as the PSV for the current element
        if (d_A[j] < d_A[idx]) {
            d_psv[idx] = j; // save just the index
            break;
        }
    }
}


__global__ void computeBlockNSV(int32_t *d_A, int32_t *d_nsv, size_t n) {
    // Calculate global thread index
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    // Check if the global thread index exceeds the bounds of the array
    if (idx >= n) return;

    // Initialize the NSV for this element to -1
    d_nsv[idx] = -1;

    // Iterate through subsequent elements to find the next smaller value
    for (int j = idx + 1; j < n; ++j) {
        // If a smaller element is found, store it as the NSV for the current element
        if (d_A[j] < d_A[idx]) {
            //d_nsv[idx] = d_A[j]; // save the val
            d_nsv[idx] = j; // save just the index

            break;
        }
    }
}



__global__ void mergeBlockPSV(int32_t *d_A, int32_t *d_psv, size_t n) {
    // Calculate global thread index
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    // Ensure that the thread index in bounds
    if (idx >= n) return;

    // Calculate the start index of the current block
    int blockStart = blockIdx.x * blockDim.x;

    // If the current block's thread did not find a PSV (indicated by d_psv[idx] == -1),
    // check in the previous blocks for a possible PSV
    if (blockIdx.x > 0 && d_psv[idx] == -1) {
        // Iterate over the elements from the end of the previous block (blockStart - 1) backwards
        for (int j = blockStart - 1; j >= 0; --j) {
            // If a smaller value is found in the previous block, assign it as the PSV
            if (d_A[j] < d_A[idx]) {
                d_psv[idx] = j;
                break;
            }
        }
    }
}


__global__ void mergeBlockNSV(int32_t *d_A, int32_t *d_nsv, size_t n) {
    // Calculate global thread index
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    // Ensure that the thread index is within bounds
    if (idx >= n) return;

    // Calculate the end index of the current block
    int blockEnd = (blockIdx.x + 1) * blockDim.x - 1;

    // If the current block's thread did not find an NSV (indicated by d_nsv[idx] == -1),
    // check in the next blocks for a possible NSV
    if (blockIdx.x < gridDim.x - 1 && d_nsv[idx] == -1) {
        // Iterate over the elements from the start of the next block (blockEnd + 1) forwards
        for (int j = blockEnd + 1; j < n; ++j) {
            // If a smaller value is found in the next block, assign it as the NSV
            if (d_A[j] < d_A[idx]) {
                d_nsv[idx] = j;
                break;
            }
        }
    }
}


__global__ void computeBlockPSVText(const int32_t *d_SA, const int32_t *d_ISA, const int32_t *d_PSVlex, int32_t *d_PSVtext, size_t n) {
    // Calculate global thread index
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    // Ensure that the thread index is within bounds
    if (idx >= n) return;

    // Compute the PSV text for the current index -- Maps PSVlex to suffix array
    if (d_PSVlex[d_ISA[idx]] == -1)
        d_PSVtext[idx] = -1;    // No PSV exists
    else
        d_PSVtext[idx] = d_SA[d_PSVlex[d_ISA[idx]]];
}


__global__ void computeBlockNSVText(const int32_t *d_SA, const int32_t *d_ISA, const int32_t *d_NSVlex, int32_t *d_NSVtext, size_t n) {
    // Calculate global thread index
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    // Ensure that the thread index is within bounds
    if (idx >= n) return;

    // Compute the NSV text for the current index -- Maps NSVlex to suffix array
    if (d_NSVlex[d_ISA[idx]] == -1)
        d_NSVtext[idx] = -1;  // No NSV exists
    else
        d_NSVtext[idx] = d_SA[d_NSVlex[d_ISA[idx]]];
}


size_t LZFactor(uint8_t *data, size_t i, int32_t psv, int32_t nsv, size_t n){
    std::pair<size_t,size_t> LZ_factor;
    size_t pos, len = 0;
    if(nsv == -1){
        while(data[psv + len] == data[i + len]) ++len;
        pos = psv;
    } else if(psv == -1) {
        while(i + len < n && data[nsv + len] == data[i + len]) ++len;
        pos = nsv;
    } else {
        while (data[psv + len] == data[nsv + len]) ++len;
        if (data[i + len] == data[psv + len]) {
            ++len;
            while (data[i + len] == data[psv + len]) ++len;
            pos = psv;
        } else {
            while (i + len < n && data[i + len] == data[nsv + len]) ++len;
            pos = nsv;
        }
    }
    if (len == 0) pos = data[i];

    std::cout << pos << " " << len << std::endl;

    return i + std::max((size_t)1, len);

    //store as bin
}

void ComputeLZ77(uint8_t *data, int32_t *d_psv_text, int32_t *d_nsv_text, size_t n){
    size_t i = 0;
    //std::vector<std::pair<size_t, size_t>> *LZ;
    while(i < n){
        i = LZFactor(data, i, d_psv_text[i + 1] - 1, d_nsv_text[i + 1] - 1, n);
    }
}


void PrintResults(auto *container, size_t length, std::string title) {
    std::cout << std::setw(26) << title;

    for (size_t i = 0; i < length; i++)
        std::cout << std::setw(5) << container[i];

    std::cout << std::endl;
}


int main(int argc, char **argv) {

    /*  Read Input Files and construct Data Vector */

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
        return 1;
    }

    std::string input_file = argv[1];
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

    size_t length = data.size();
    uint32_t *SA  = new uint32_t[length];
    uint32_t *ISA = new uint32_t[length];

    // allocate device storage
    void *device_storage = nullptr;
    int64_t result = libcubwt_allocate_device_storage(&device_storage, length);
    if (result != LIBCUBWT_NO_ERROR) {
        printf("Allocation error: %ld\n", result);
        return 1;
    }

    // generate suffix array
    result = libcubwt_sa_isa(device_storage, data.data(), SA, ISA, length);
    if (result != LIBCUBWT_NO_ERROR) {
        printf("Suffix array error: %ld\n", result);
        return 1;
    }


    // add printout here for time of SA generation



    /* Create PSV and NSV lexicographical */

    // Declare device pointers for the suffix array and PSV array
    int32_t *d_SA, *d_psv, *d_nsv;

    // Allocate memory on the GPU for the suffix array and PSV array
    cudaMalloc((void**)&d_SA,  length * sizeof(int32_t));
    cudaMalloc((void**)&d_psv, length * sizeof(int32_t));
    cudaMalloc((void**)&d_nsv, length * sizeof(int32_t));

    // Copy the host's suffix array to the device memory
    cudaMemcpy(d_SA, SA, length * sizeof(int32_t), cudaMemcpyHostToDevice);

    // Set block and grid sizes for kernel execution
    int blockSize = 256;
    int numBlocks = (length + blockSize - 1) / blockSize;

    // Launch kernel to compute block-level PSV/NSV (lexicographical) in parallel
    computeBlockPSV<<<numBlocks, blockSize>>>(d_SA, d_psv, length);
    computeBlockNSV<<<numBlocks, blockSize>>>(d_SA, d_nsv, length);
    cudaDeviceSynchronize();  // Ensure all threads have finished executing before continuing

    // Merge block-level PSV results to finalize PSV/NSV values across blocks
    mergeBlockPSV<<<numBlocks, blockSize>>>(d_SA, d_psv, length);
    mergeBlockNSV<<<numBlocks, blockSize>>>(d_SA, d_nsv, length);
    cudaDeviceSynchronize();  // Synchronize again to ensure all threads complete



    /* Compute the PSV and NSV text */

    // Declare device pointers for the inverse suffix array and PSV/NSV text arrays
    int32_t *d_ISA, *d_psv_text, *d_nsv_text;

    // Allocate memory on the GPU for the inverse suffix array and PSV/NSV text arrays
    cudaMalloc((void**)&d_ISA,      length * sizeof(int32_t));
    cudaMalloc((void**)&d_psv_text, length * sizeof(int32_t));
    cudaMalloc((void**)&d_nsv_text, length * sizeof(int32_t));

    // Copy the host's inverse suffix array to the device memory
    cudaMemcpy(d_ISA, ISA, length * sizeof(int32_t), cudaMemcpyHostToDevice);

    // Launch kernel to compute block-level PSV/NSV text in parallel
    computeBlockPSVText<<<numBlocks, blockSize>>>(d_SA, d_ISA, d_psv, d_psv_text, length);
    computeBlockNSVText<<<numBlocks, blockSize>>>(d_SA, d_ISA, d_nsv, d_nsv_text, length);
    cudaDeviceSynchronize();



    /* Bring everything of relevance back to device memory and free unused memory on GPU */

    // Initialize PSV/NSV vector with a default value of -1 for each element
    std::vector<int32_t> psv_lex(length, -1);
    std::vector<int32_t> nsv_lex(length, -1);
    std::vector<int32_t> psv_text(length, -1);
    std::vector<int32_t> nsv_text(length, -1);

    // Copy the computed PSV array from device memory back to the host memory
    cudaMemcpy(psv_lex.data(), d_psv, length * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(nsv_lex.data(), d_nsv, length * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(psv_text.data(), d_psv_text, length * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(nsv_text.data(), d_nsv_text, length * sizeof(int32_t), cudaMemcpyDeviceToHost);

    // Free the allocated memory on the GPU to avoid memory leaks
    cudaFree(d_SA);
    cudaFree(d_ISA);
    cudaFree(d_psv);
    cudaFree(d_nsv);
    cudaFree(d_nsv_text);
    cudaFree(d_psv_text);



    /* Compute the LZ77 data structure */

    // Declare device pointer for the LZ77 output data
    //size_t *d_LZ77_p, d_LZ77_l;

    // Allocate memory on the GPU for the inverse suffix array and PSV/NSV text arrays
    //cudaMalloc((void**)&d_LZ77_p, length * sizeof(size_t));
    //cudaMalloc((void**)&d_LZ77_l, length * sizeof(size_t));

    /* Print/Save Results */

    // Print original String
    PrintResults(data.data(), length, "Input Data:");

    // Print Index Locations
    std::vector<int> index(length);
    std::iota(index.begin(), index.end(), 0);
    PrintResults(index.data(), length, "Index Locations:");
    std::cout << "\n";

    // Print SA output
    PrintResults(SA, length, "Suffix Array:");
    PrintResults(ISA, length, "Inv Suffix Array:");
    std::cout << "\n";

    // Print PSV/NSV output
    PrintResults(psv_lex.data(), length, "PSV Lex:");
    PrintResults(nsv_lex.data(), length, "NSV Lex:");
    std::cout << "\n";

    PrintResults(psv_text.data(), length, "PSV Text:");
    PrintResults(nsv_text.data(), length, "NSV Text:");
    std::cout << "\n";

    ComputeLZ77(data.data(), psv_text.data(), nsv_text.data(), length);


    /*
    for(size_t i = 0; i < length; ++i){
        //std::cout << psv[i] << " ";
        std::cout << nsv[i] << " ";
    } */

    // free device storage
    delete[] SA;

    return 0;
}



