#include "handle_error.cuh"
#include "sq.cuh"
#include <cuda_runtime.h>

__global__ void sq_kernel(int n, uint64_t *input) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    input[i] = 1ULL << input[i];
  }
}

void sq(int N, uint64_t *input) {
  int blockSize = 64;
  int numBlocks = (N + blockSize - 1) / blockSize;

  sq_kernel<<<numBlocks, blockSize>>>(N, input);
  handleCUDAErrors();
}