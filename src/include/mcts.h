#pragma once

#include "move_gen.h"
#include "node.h"
#include "predictor.h"

struct GlobalData {
  uint32_t currBatchNum = 0;
  // This game's pending evaluations for the current collect epoch (one wave
  // in flight per game at a time). A worker pushes newly found unevaluated
  // leaves here as it discovers them; grouping them into forward passes is
  // decided by the central predictor, never by the game.
  std::vector<std::shared_ptr<Request>> requests;
  Predictor *predictor;
  [[maybe_unused]] const torch::Device device;
  // Runtime-tunable search budget; defaults to constants.h SIMULATIONS.
  int simulations = SIMULATIONS;

  GlobalData(const torch::Device &_device, Predictor *_predictor)
      : predictor(_predictor), device(_device) {};
};

static constexpr bool POLICY_IS_LOGITS = true;
struct Sample {
  torch::Tensor input;
  std::vector<std::pair<int, float>> policy;
  float value = 0.0f;
};

Sample makeSample(Node *root);
float terminalValue(Midnight::Position &board);

Node *getNextMove(Node *node, float temperature, GlobalData &g);
float terminalValue(Midnight::Position &board);
bool isTerminal(Midnight::Position &board);
int policyIndex(Midnight::Position &board, Midnight::Move move);
std::vector<Midnight::Move> createMovelistVec(Midnight::Position board);
