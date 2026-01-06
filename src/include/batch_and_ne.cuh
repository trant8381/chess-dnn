#pragma once
#include <cstdint>

void batch_and_ne(int numBatches, uint64_t *mask, uint64_t *batch,
                  float *result);