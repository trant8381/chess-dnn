#pragma once

#include "move_gen.h"
#include <cmath>
#include <cstdint>
#include <set>
#include <torch/torch.h>

constexpr int HISTORY_BOARDS = 4; // model's amount of history positions stored.
constexpr int INPUT_PLANES =
    HISTORY_BOARDS * 14 +
    7; // (6 white pieces + 6 black pieces + 2 repetitions) per history board. 7
       // situational planes.
constexpr int TRUNK_CHANNELS = 64; // channels per resnet block.
constexpr int TOWER_SIZE = 6;      // amount of resnet blocks.
constexpr float C_PUCT = 1.5f;     // PUCT constant for MCTS selection.
constexpr int SIMULATIONS = 10;   // amount of simulations for one move.
constexpr float FPU = -0.2f;       // temperature constant for move selection.
constexpr uint64_t TABLE_SIZE = 1ULL << 25; // size of transposition table.
constexpr float UNKNOWN = INFINITY;         // unknown value for batchPUCT.
constexpr float VL = 2;         // virtual loss value for updating statistics.


// the return struct on a forward pass of the whole model.
struct Eval {
  torch::Tensor value;
  torch::Tensor policy;
  bool initialized = false;
  Eval();
  Eval(torch::Tensor _value, torch::Tensor _policy) {
    value = _value;
    policy = _policy;
    initialized = true;
  }
};

struct Statistics {
  float *totalValue;
  uint32_t *visitCount;

  Statistics(float *_totalValue, uint32_t *_visitCount)
      : totalValue(_totalValue), visitCount(_visitCount) {}
};

// one node in the mcts game tree.
struct Node {
  Node *parent;
  std::set<Node *> children;
  Midnight::Position position;

  float totalValue = 0;
  uint32_t visitCount = 0;
  bool initialized = false;

  float batch_totalValue = 0;
  uint32_t batch_visitCount = 0;
  bool batch_initialized = false;

  int depth;

  float valueEval;
  float policyEval;

  Node(Node *_parent, const std::set<Node *> _children,
       const Midnight::Position _position) {
    parent = _parent;
    children = _children;
    position = _position;
  }

  Node(Node *_parent, const std::set<Node *> _children,
       const Midnight::Position _position, float _policy) {
    parent = _parent;
    children = _children;
    position = _position;
    policyEval = _policy;
  }

  void reinitializeBatch() {
    batch_totalValue = totalValue;
    batch_visitCount = visitCount;
  }

  Statistics getTreeStats(bool isBatch) {
    if (isBatch) {
      return Statistics(&batch_totalValue, &batch_visitCount);
    } else {
      return Statistics(&totalValue, &visitCount);
    }
  }

  ~Node() {
    for (Node *node : children) {
      delete node;
    }
  }
};