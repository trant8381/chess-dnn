#pragma once

#include "dnn.h"
#include "move_gen.h"
#include "node.h"

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

struct Sample {
  torch::Tensor input;
  std::vector<std::pair<int, float>> policy;
  float value = 0.0f;
};

Sample makeSample(Node *root);
float terminalValue(Midnight::Position &board);

Node *getNextMove(Node *node, DNN &model, float temperature, GlobalData &g);
float terminalValue(Midnight::Position &board);
bool isTerminal(Midnight::Position &board);
int policyIndex(Midnight::Position &board, Midnight::Move move);
std::vector<Midnight::Move> createMovelistVec(Midnight::Position board);