//
// Created by qin on 4/15/22.
//

#ifndef TEST__CUDA_TEST_CUH_
#define TEST__CUDA_TEST_CUH_

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

__forceinline__ __device__ float abs_value(float in){
  return fabs(in);
}

#endif //TEST__CUDA_TEST_CUH_
