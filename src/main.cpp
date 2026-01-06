#include "constants.h"
#include "create_state.h"
#include "ctpl.h"
#include "dnn.h"
#include "mcts.h"
#include "move_gen.h"
#include <ATen/Context.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <cstddef>
#include <torch/cuda.h>
#include <torch/torch.h>

struct State {
  Midnight::Position position;
  float value;
};

void playGame(Node *root, DNN &model, const torch::Device device) {
  GlobalData g = GlobalData(device);
  while (true) {
    if (isTerminal(root->position)) {
      break;
    }
    float temperature = 1.0f;
    Node *selected = getNextMove(root, model, temperature, g);

    for (Node *node : root->children) {
      if (node != selected) {
        delete node;
      }
    }
    root->children.clear();
    root->children.insert(selected);

    root = selected;

    temperature = std::pow(temperature + 1, TEMPERATURE_DECAY);

    std::cout << root->position << std::endl;
  }

  while (root) {
    std::cout << root->position.fen() << std::endl;
    root = root->parent;
  }
}

Node *createRoot() {
  return new Node(nullptr, {}, Midnight::Position(Midnight::START_FEN));
}

int main() {
  ctpl::thread_pool pool(PARALLEL_GAMES);
  for (size_t i = 0; i < PARALLEL_GAMES; i++) {
    pool.push([i](int) {
      Node *root = createRoot();
      torch::Device device = torch::kCPU;
      if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA, i % torch::getNumGPUs());
      }

      DNN model = DNN();
      torch::NoGradGuard no_grad;
      model->to(device);

      playGame(root, model, device);
    });
  }

  return 0;
}