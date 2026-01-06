#pragma once

#include "dnn.h"
#include "move_gen.h"
#include <cmath>

// one node in the mcts game tree.
struct Node {
  Node *parent;
  std::set<Node *> children;
  Midnight::Position position;

  float totalValue = 0;
  uint32_t visitCount = 0;

  float batch_totalValue = 0;
  uint32_t batch_visitCount = 0;

  bool initialized = false;

  uint32_t batchNum = 0;

  float valueEval = INFINITY;
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

  ~Node() {
    for (Node *node : children) {
      delete node;
    }
  }
};

struct Batch {
  std::vector<torch::Tensor> nnInputs;
  std::vector<Node *> nodes;
};

struct GlobalData {
  uint32_t currBatchNum = 0;
  Batch batch = {};
  const torch::Device device;

  GlobalData(const torch::Device &_device) : device(_device) {};
};

Node *getNextMove(Node *node, DNN &model, float temperature, GlobalData &g);
bool isTerminal(Midnight::Position &board);