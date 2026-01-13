#pragma once

#include "concurrent_queue.h"
#include "dnn.h"
#include "move_gen.h"
#include "node.h"
#include <cmath>
#include <cstdint>

struct Batch {
  std::vector<torch::Tensor> nnInputs;
  std::vector<Node *> nodes;
};

struct GlobalData {
  uint16_t simulation = 0;
  uint32_t currBatchNum = 0;
  Batch batch = {};
  torch::Device device = torch::kCPU;
  moodycamel::ConcurrentQueue<Node*> *q;

  GlobalData() = default;
  GlobalData(const torch::Device &_device, moodycamel::ConcurrentQueue<Node*>* _q) : device(_device), q(_q) {};
};

void putBatch(Node *node, Eval &outputs, GlobalData &g);
Node *getNextMove(Node *node, DNN &model, float temperature, GlobalData &g);
bool isTerminal(Midnight::Position &board);