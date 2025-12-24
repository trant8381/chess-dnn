#pragma once

#include <torch/torch.h>

constexpr int HISTORY_BOARDS = 4; // Model's amount of history positions stored.
constexpr int INPUT_PLANES =
    HISTORY_BOARDS * 14 +
    7; // (6 white pieces + 6 black pieces + 2 repetitions) per history board. 7
       // situational planes.
constexpr int TRUNK_CHANNELS = 256; // channels per resnet block.
constexpr int TOWER_SIZE = 20;      // amount of resnet blocks.

// the return struct on a forward pass of the whole model.
struct Eval {
  torch::Tensor value;
  torch::Tensor policy;

  Eval(torch::Tensor _value, torch::Tensor _policy) {
    value = _value;
    policy = _policy;
  }
};