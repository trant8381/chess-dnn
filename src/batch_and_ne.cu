#include "batch_and_ne.h"
#include "constants.h"
#include "handle_error.cuh"
#include <cuda_runtime.h>

__global__ void batch_and_ne_kernel(int numBatches, uint64_t *mask,
                                    uint64_t *batch, float *result) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < numBatches * HISTORY_BOARDS * 14 * 8 * 8) {
    result[i] = (mask[i % 64] & batch[i / 64]) != 0;
  }
}

void batch_and_ne(int numBatches, uint64_t *mask, uint64_t *batch,
                  float *result) {
  int blockSize = 64;
  int numBlocks =
      (numBatches * HISTORY_BOARDS * 14 * 8 * 8 + blockSize - 1) / blockSize;
  batch_and_ne_kernel<<<numBlocks, blockSize>>>(numBatches, mask, batch,
                                                result);
  handleCUDAErrors();
}