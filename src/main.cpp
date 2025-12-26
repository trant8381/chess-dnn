#include "dnn.h"
#include "mcts.h"
#include "move_gen.h"
#include <torch/torch.h>

int main() {
  DNN model = DNN();

  Node root = Node({}, Midnight::Position(Midnight::START_FEN), 0);

  for (int i = 0; i < 800; i++) {
    simulate(&root, model);
  }

  for (Node &child : root.children) {
    std::cout << child.meanValue << std::endl;
  }
}