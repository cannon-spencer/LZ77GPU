//
// Created by yu hong on 11/17/24.
//

#ifndef LIBCUBWT_KERNEL_CUH
#define LIBCUBWT_KERNEL_CUH

#include "data_types.cuh"

__global__ void blockPSVNSVKernel(const Element* input, Element* psv_output,
                                  Element* nsv_output, const int blockSize,
                                  const int totalSize);

#endif //LIBCUBWT_KERNEL_CUH
