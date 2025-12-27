#pragma once

#include <torch/torch.h>

constexpr int HISTORY_BOARDS = 4; // Model's amount of history positions stored.
constexpr int INPUT_PLANES =
    HISTORY_BOARDS * 14 +
    7; // (6 white pieces + 6 black pieces + 2 repetitions) per history board. 7
       // situational planes.
constexpr int TRUNK_CHANNELS = 64;  // channels per resnet block.
constexpr int TOWER_SIZE = 6;       // amount of resnet blocks.
constexpr float C_PUCT = 1.5f;      // PUCT constant for MCTS selection.
constexpr int SIMULATIONS = 200;    // amount of simulations for one move.
constexpr float TEMPERATURE = 1.0f; // temperature constant for move selection

// the return struct on a forward pass of the whole model.
struct Eval {
  torch::Tensor value;
  torch::Tensor policy;

  Eval(torch::Tensor _value, torch::Tensor _policy) {
    value = _value;
    policy = _policy;
  }
};