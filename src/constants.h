#pragma once

#include <torch/torch.h>

constexpr int HISTORY_BOARDS = 4;
constexpr int INPUT_PLANES = 4 * 14 + 7;
constexpr int TRUNK_CHANNELS = 256;
constexpr int TOWER_SIZE = 20;

struct Eval {
  torch::Tensor value;
  torch::Tensor policy;

  Eval(torch::Tensor _value, torch::Tensor _policy) {
    value = _value;
    policy = _policy;
  }
};