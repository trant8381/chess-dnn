#pragma once

#include "dnn.h"
#include "move_gen.h"
#include "node.h"
#include <cmath>

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