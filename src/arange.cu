#include "arange.cuh"
#include "handle_error.cuh"
#include <cuda_runtime.h>

__global__ void arange_kernel(uint64_t *arr, int n, float start, float step) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    arr[i] = start + i * step;
  }
}

void arange(int N, uint64_t *result) {
  int blockSize = 64;
  int numBlocks = (N + blockSize - 1) / blockSize;

  arange_kernel<<<numBlocks, blockSize>>>(result, N, 0.0f, 1.0f);
  handleCUDAErrors();
}