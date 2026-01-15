#include "node.h"
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

NNInputBatch constructHistoryFast(Node** &begin, Node** &end);
torch::Tensor createStateFast(Node** begin, Node** end,
                              const torch::Device device);