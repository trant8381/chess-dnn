#include "mcts.h"
#include "move_gen.h"
#include "constants.h"
#include <torch/torch.h>
#include <vector>

const Midnight::Position START_POS = Midnight::Position(Midnight::START_FEN);
typedef std::vector<std::array<uint64_t, HISTORY_BOARDS * 14>> Histories;

struct NNInputBatch {
  Histories histories;
  std::vector<std::array<float, 7>> scalars;
};

Histories constructHistoryFast(Node *node);
torch::Tensor createStateFast(const std::vector<Node *> &nodes,
                              const torch::Device &device);