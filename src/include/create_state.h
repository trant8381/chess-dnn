#pragma once

#include "constants.h"
#include "move_gen.h"
#include "mcts.h"
#include <ATen/core/TensorBody.h>
#include <torch/torch.h>

using namespace Midnight;
typedef std::vector<std::array<uint64_t, HISTORY_BOARDS * 14>> Histories;

struct NNInputBatch {
  Histories histories;
  std::vector<std::array<float, 7>> scalars;
};

std::array<Position, HISTORY_BOARDS> constructHistory(Node *node);
torch::Tensor createState(const std::array<Position, HISTORY_BOARDS> &boards,
                          const torch::Device &device);
Histories constructHistoryFast(Node *node);
torch::Tensor createStateFast(const std::vector<Node*> &nodes, const torch::Device &device);