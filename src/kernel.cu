//
// Created by yu hong on 11/17/24.
//

#include "kernel.cuh"

__global__ void blockPSVNSVKernel(const Element* input, Element* psv_output,
                                 Element* nsv_output, const int blockSize,
                                 const int totalSize) {
    __shared__ Element psv_stack[1024];
    __shared__ Element nsv_stack[1024];
    __shared__ int psv_stackSize, nsv_stackSize;

    // compute block start and current block size
    int blockStart = blockIdx.x * blockSize;
    int currentBlockSize = min(blockSize, totalSize - blockStart);

    // initialize stack size
    if (threadIdx.x == 0) {
        psv_stackSize = 0;
        nsv_stackSize = 0;
    }
    __syncthreads();

    // compute PSV from left to right
    for (int i = 0; i < currentBlockSize; i++) {
        __syncthreads();
        if (threadIdx.x == 0) {
            Element current = input[blockStart + i];

            while (psv_stackSize > 0 && psv_stack[psv_stackSize-1].value >= current.value) {
                psv_stackSize--;
            }

            if (psv_stackSize > 0) {
                psv_output[blockStart + i] = psv_stack[psv_stackSize-1];
            } else {
                psv_output[blockStart + i] = Element(INT_MAX, -1);
            }

            psv_stack[psv_stackSize++] = current;
        }
    }

    // compute NSV from right to left
    for (int i = currentBlockSize - 1; i >= 0; i--) {
        __syncthreads();
        if (threadIdx.x == 0) {
            Element current = input[blockStart + i];

            while (nsv_stackSize > 0 && nsv_stack[nsv_stackSize-1].value >= current.value) {
                nsv_stackSize--;
            }

            if (nsv_stackSize > 0) {
                nsv_output[blockStart + i] = nsv_stack[nsv_stackSize-1];
            } else {
                nsv_output[blockStart + i] = Element(INT_MAX, -1);
            }

            nsv_stack[nsv_stackSize++] = current;
        }
    }
}
