#include "dnn.h"
#include "mcts.h"
#include "move_gen.h"
#include <torch/torch.h>

int main() {
  DNN model = DNN();

  Node root = Node({}, Midnight::Position(Midnight::START_FEN), 0);

  while (true) {
    if (isTerminal(root.position)) {
      break;
    }
    float temperature = 1.0f;
    root = playMove(&root, model, temperature);
    temperature = std::pow(temperature + 1, -0.42f);

    std::cout << root.position << std::endl;
  }
}