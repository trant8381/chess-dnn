#pragma once

#include "constants.h"
#include "mcts.h"
#include "move_gen.h"
#include <ATen/core/TensorBody.h>
#include <torch/torch.h>

using namespace Midnight;

std::array<Position, HISTORY_BOARDS> constructHistory(Node *node);
torch::Tensor createState(const std::array<Position, HISTORY_BOARDS> &boards,
                          const torch::Device &device);