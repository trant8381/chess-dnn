#include "mcts.h"
#include "constants.h"
#include <torch/torch.h>
#include <vector>

struct NNInputBatch {
  Histories histories;
  std::vector<std::array<float, 7>> scalars;
};

Histories constructHistoryFast(Node *node);
torch::Tensor createStateFast(const std::vector<Node *> &nodes,
                              const torch::Device &device);