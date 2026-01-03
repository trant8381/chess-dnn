#include "constants.h"
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

void playGame(Node *root, DNN &model, torch::Device device) {
  GlobalData g;
  while (true) {
    if (isTerminal(root->position)) {
      break;
    }
    float temperature = 1.0f;
    Node *selected = getNextMove(root, model, device, temperature, g);

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
  std::vector<torch::Device> devices = {torch::kCPU};
  std::vector<DNN> models = {DNN()};
  std::vector<Node *> rootNodes = {createRoot()};

  if (torch::cuda::is_available()) {
    devices = {};
    models = {};
    rootNodes = {};
    for (size_t i = 0; i < torch::getNumGPUs(); i++) {
      devices.push_back(torch::Device(torch::kCUDA, i));
      models.push_back(DNN());
      rootNodes.push_back(createRoot());
    }
  }

  torch::NoGradGuard no_grad;
  ctpl::thread_pool pool(PARALLEL_GAMES);
  std::vector<Node *> roots;

  for (size_t i = 0; i < PARALLEL_GAMES; i++) {
    Node *root = createRoot();
    roots.push_back(root);

    DNN &model = models[i % devices.size()];
    model->eval();
    auto function = [&](int id) {
      (void)id;
      playGame(root, model, devices[i % devices.size()]);
    };
    pool.push(function);
  }

  return 0;
}