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

#define BLOCK_SIZE 256
#define STACK_SIZE BLOCK_SIZE

//__global__ void computePSVAndNSV(int32_t *d_A, int32_t *d_psv, int32_t *d_nsv, size_t n) {
//    __shared__ int32_t s_data[BLOCK_SIZE];
//    __shared__ int32_t s_stack[STACK_SIZE];
//    __shared__ int32_t s_stackSize;
//
//    int tid = threadIdx.x;
//    int gid = blockIdx.x * blockDim.x + threadIdx.x;
//
//    if (gid < n) {
//        s_data[tid] = d_A[gid];
//    }
//
//    if (tid == 0) {
//        s_stackSize = 0;
//    }
//    __syncthreads();
//
//    if (gid < n) {
//        d_psv[gid] = -1;
//        d_nsv[gid] = -1;
//    }
//
//    if (tid == 0) {
//        for (int i = 0; i < BLOCK_SIZE && (blockIdx.x * blockDim.x + i) < n; i++) {
//            while (s_stackSize > 0 && s_data[s_stack[s_stackSize-1]] >= s_data[i]) {
//                s_stackSize--;
//            }
//
//            if (s_stackSize > 0) {
//                d_psv[blockIdx.x * blockDim.x + i] = blockIdx.x * blockDim.x + s_stack[s_stackSize-1];
//            }
//
//            s_stack[s_stackSize++] = i;
//        }
//    }
//    __syncthreads();
//
//    if (tid == 0) {
//        s_stackSize = 0;
//    }
//    __syncthreads();
//
//    if (tid == 0) {
//        int blockEnd = min(BLOCK_SIZE, (int)(n - blockIdx.x * blockDim.x));
//
//        for (int i = blockEnd - 1; i >= 0; i--) {
//            while (s_stackSize > 0 && s_data[s_stack[s_stackSize-1]] >= s_data[i]) {
//                s_stackSize--;
//            }
//
//            if (s_stackSize > 0) {
//                d_nsv[blockIdx.x * blockDim.x + i] = blockIdx.x * blockDim.x + s_stack[s_stackSize-1];
//            }
//
//            s_stack[s_stackSize++] = i;
//        }
//    }
//}

__global__ void computePSVAndNSV(int32_t *d_A, int32_t *d_psv, int32_t *d_nsv, size_t n) {
    // Calculate global thread index
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    // Check if the global thread index exceeds the bounds of the array
    if (idx >= n) return;

    // Initialize the NSV for this element to -1
    d_psv[idx] = -1;
    d_nsv[idx] = -1;

    // Iterate through previous elements to find the previous smaller value
    for (int j = idx - 1; j >= 0; --j) {
        // If a smaller element is found, store it as the PSV for the current element
        if (d_A[j] < d_A[idx]) {
            d_psv[idx] = j; // save just the index
            break;
        }
    }

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


__global__ void mergeBlockResults(int32_t *d_A, int32_t *d_psv, int32_t *d_nsv, size_t n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid >= n) return;

    if (d_psv[gid] == -1 && blockIdx.x > 0) {
        for (int j = blockIdx.x * blockDim.x - 1; j >= 0; j--) {
            if (d_A[j] < d_A[gid]) {
                d_psv[gid] = j;
                break;
            }
        }
    }

    if (d_nsv[gid] == -1 && blockIdx.x < gridDim.x - 1) {
        for (int j = (blockIdx.x + 1) * blockDim.x; j < n; j++) {
            if (d_A[j] < d_A[gid]) {
                d_nsv[gid] = j;
                break;
            }
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

__global__ void computePSVNSVText(const int32_t *d_SA, const int32_t *d_ISA,
                                       const int32_t *d_PSVlex, const int32_t *d_NSVlex,
                                       int32_t *d_PSVtext, int32_t *d_NSVtext, size_t n) {
    extern __shared__ int32_t shared_ISA[];

    // Calculate global thread index
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int tid = threadIdx.x;

    if (idx < n) {
        shared_ISA[tid] = d_ISA[idx];
    }
    __syncthreads();

    // Ensure that the thread index is within bounds
    if (idx >= n) return;

    int32_t isa_val = shared_ISA[tid];

    if (isa_val < n) {  // Ensure isa_val is within bounds
        d_PSVtext[idx] = (d_PSVlex[isa_val] == -1) ? -1 : d_SA[d_PSVlex[isa_val]];
        d_NSVtext[idx] = (d_NSVlex[isa_val] == -1) ? -1 : d_SA[d_NSVlex[isa_val]];
    } else {
        d_PSVtext[idx] = -1;
        d_NSVtext[idx] = -1;
    }
}


// CODE FROM KKP3
size_t LZFactor(uint8_t *data, size_t i, int32_t psv, int32_t nsv, size_t n, size_t &pos, size_t &len){

    len = 0;

    auto matchLength = [&](int32_t baseIdx) {
        size_t l = 0;
        while (i + l < n && baseIdx + l < n && data[baseIdx + l] == data[i + l]) {
            ++l;
        }
        return l;
    };

    // Case 1: If there's no next smaller value (NSV), only consider the previous smaller value (PSV)
    if(nsv == -1){
        // Continue matching as long as characters from PSV match the target position in 'data'
        len = matchLength(psv);
        pos = psv; // Set position to PSV since NSV is invalid
    }

    // Case 2: If there's no previous smaller value (PSV), only consider the next smaller value (NSV)
    else if(psv == -1) {
        // Continue matching as long as characters from NSV match the target position in 'data'
        len = matchLength(nsv);
        pos = nsv; // Set position to NSV since PSV is invalid
    }

    // Case 3: Both PSV and NSV are valid, so we need to find the best match between the two
    else {
        // First, match the common length between PSV and NSV in 'data'
        size_t commonLen = 0;
        // Match common length for both PSV and NSV
        while (i + commonLen < n && psv + commonLen < n && nsv + commonLen < n &&
               data[psv + commonLen] == data[nsv + commonLen]) {
            ++commonLen;
        }
        len = commonLen;


        // Extend match using the best option between PSV and NSV
        if (i + len < n && data[i + len] == data[psv + len]) {
            ++len;
            len += matchLength(psv + len);
            pos = psv;
        } else {
            len += matchLength(nsv + len);
            pos = nsv;
        }
    }

    // If no match was found, set the position to the current data character
    if (len == 0) pos = data[i];

    // send output to binary file
    return i + std::max((size_t) 1, len);

}

void ComputeLZ77(uint8_t *data, int32_t *d_psv_text, int32_t *d_nsv_text, size_t n, std::string file_name){
    size_t i = 0;

    std::vector<std::pair<size_t, size_t>> buffer;

    while(i < n){
        size_t pos, len;
        i = LZFactor(data, i, d_psv_text[i + 1] - 1, d_nsv_text[i + 1] - 1, n, pos, len);
        buffer.push_back(std::make_pair(pos, len));
    }
    printf("LZ77 compression successful\n");

    // Write all results to the output file at once
    std::ofstream out_file(file_name, std::ios::binary);
    for (const auto &lz : buffer) {
        out_file.write(reinterpret_cast<const char*>(&lz.first), sizeof(size_t));
        out_file.write(reinterpret_cast<const char*>(&lz.second), sizeof(size_t));
    }

    out_file.close();
}


void PrintResults(auto *container, size_t length, std::string title) {
    std::cout << std::setw(26) << title;

    for (size_t i = 0; i < length; i++)
        std::cout << std::setw(5) << container[i];

    std::cout << std::endl;
}

void SaveArrayToFile(const std::vector<int32_t>& array, const std::string& filename) {
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        return;
    }

    // Write the size of the array first
    size_t size = array.size();
    outFile.write(reinterpret_cast<const char*>(&size), sizeof(size_t));

    // Write the array data
    outFile.write(reinterpret_cast<const char*>(array.data()), size * sizeof(int32_t));

    outFile.close();
    std::cout << "Successfully wrote array to " << filename << std::endl;
}

void SavePSVNSVText(const std::vector<int32_t>& psv_text,
                    const std::vector<int32_t>& nsv_text,
                    const std::string& input_file) {
    // Create filenames based on input file
    std::string psv_filename = input_file + ".psv";
    std::string nsv_filename = input_file + ".nsv";

    // Save PSV and NSV arrays
    SaveArrayToFile(psv_text, psv_filename);
    SaveArrayToFile(nsv_text, nsv_filename);
}



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

    // Record start event
    cudaEventRecord(start);

    // generate suffix array
    result = libcubwt_sa_isa(device_storage, data.data(), SA, ISA, length);
    if (result != LIBCUBWT_NO_ERROR) {
        printf("Suffix array error: %ld\n", result);
        return 1;
    }

    // Record stop event and calculate elapsed time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time for computing SA and ISA: " << milliseconds << " ms" << std::endl;

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
    int blockSize = BLOCK_SIZE;
    int numBlocks = (length + blockSize - 1) / blockSize;

    // Record start event
    cudaEventRecord(start);

    // Launch kernel to compute block-level PSV/NSV (lexicographical) in parallel
    computePSVAndNSV<<<numBlocks, blockSize>>>(d_SA, d_psv, d_nsv, length);
    cudaDeviceSynchronize();  // Ensure all threads have finished executing before continuing

    // Merge block-level PSV results to finalize PSV/NSV values across blocks
//    mergeBlockResults<<<numBlocks, blockSize>>>(d_SA, d_psv, d_nsv, length);
    mergeBlockPSV<<<numBlocks, blockSize>>>(d_SA, d_psv, length);
    mergeBlockNSV<<<numBlocks, blockSize>>>(d_SA, d_nsv, length);
    cudaDeviceSynchronize();  // Synchronize again to ensure all threads complete

    // Record stop event and calculate elapsed time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time for computing PSV and NSV: " << milliseconds << " ms" << std::endl;

    // print the PSV and NSV values
//    std::vector<int32_t> psv_lex(length);
//    std::vector<int32_t> nsv_lex(length);
//    cudaMemcpy(psv_lex.data(), d_psv, length * sizeof(int32_t), cudaMemcpyDeviceToHost);
//    cudaMemcpy(nsv_lex.data(), d_nsv, length * sizeof(int32_t), cudaMemcpyDeviceToHost);
//
//    PrintResults(psv_lex.data(), length, "PSV Lexicographical:");
//    PrintResults(nsv_lex.data(), length, "NSV Lexicographical:");
//    std::cout << "PSV and NSV constrcuted successful "<<std::endl;



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
    size_t sharedMemSize = blockSize * sizeof(int32_t);

    // Record start event
    cudaEventRecord(start);

    computePSVNSVText<<<numBlocks, blockSize, sharedMemSize>>>(d_SA, d_ISA, d_psv, d_nsv, d_psv_text, d_nsv_text, length);
    cudaDeviceSynchronize();

    // Record stop event and calculate elapsed time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time for computing PSV and NSV text: " << milliseconds << " ms" << std::endl;

    // print the PSV and NSV values
//    std::vector<int32_t> psv_test(length);
//    std::vector<int32_t> nsv_test(length);
//    cudaMemcpy(psv_test.data(), d_psv_text, length * sizeof(int32_t), cudaMemcpyDeviceToHost);
//    cudaMemcpy(nsv_test.data(), d_nsv_text, length * sizeof(int32_t), cudaMemcpyDeviceToHost);
//
//    PrintResults(psv_test.data(), length, "PSV Text:");
//    PrintResults(nsv_test.data(), length, "NSV Text:");
    std::cout << "PSV and NSV text constrcuted successful "<<std::endl;



    /* Bring everything of relevance back to device memory and free unused memory on GPU */

    // Initialize PSV/NSV vector with a default value of -1 for each element
//    std::vector<int32_t> psv_lex(length, -1);
//    std::vector<int32_t> nsv_lex(length, -1);
    std::vector<int32_t> psv_text(length, -1);
    std::vector<int32_t> nsv_text(length, -1);

    // Copy the computed PSV array from device memory back to the host memory
//    cudaMemcpy(psv_lex.data(), d_psv, length * sizeof(int32_t), cudaMemcpyDeviceToHost);
//    cudaMemcpy(nsv_lex.data(), d_nsv, length * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(psv_text.data(), d_psv_text, length * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(nsv_text.data(), d_nsv_text, length * sizeof(int32_t), cudaMemcpyDeviceToHost);

    // Free the allocated memory on the GPU to avoid memory leaks
    cudaFree(d_SA);
    cudaFree(d_ISA);
    cudaFree(d_psv);
    cudaFree(d_nsv);
    cudaFree(d_nsv_text);
    cudaFree(d_psv_text);
    SavePSVNSVText(psv_text, nsv_text, input_file);
    ComputeLZ77(data.data(), psv_text.data(), nsv_text.data(), length, output_file);

    delete[] SA;

    return 0;
}



