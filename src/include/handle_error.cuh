#pragma once

#include <cuda_runtime.h>
#include <iostream>

inline void handleCUDAErrors() {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "Kernel launch error: " << cudaGetErrorString(err)
              << std::endl;
  }
}