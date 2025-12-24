#pragma once

#include "constants.h"
#include "move_gen.h"
#include <ATen/core/TensorBody.h>
#include <torch/torch.h>

using namespace Midnight;

torch::Tensor createState(const std::array<Position, HISTORY_BOARDS> &boards);