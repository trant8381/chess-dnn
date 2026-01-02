#include "dnn.h"
#include "mcts.h"
#include "move_gen.h"
#include <ATen/Context.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <torch/cuda.h>
#include <torch/torch.h>

int main() {
  DNN model = DNN();
  torch::Device device = torch::kCPU;
  if (torch::cuda::is_available()) {
    device = torch::kCUDA;
  }

  model->to(device, 0);
  torch::NoGradGuard no_grad;
  model->eval();

  Node *root = new Node(nullptr, {}, Midnight::Position(Midnight::START_FEN));

  while (true) {
    if (isTerminal(root->position)) {
      break;
    }
    float temperature = 1.0f;
    Node* selected = getNextMove(root, model, device, temperature);

    for (Node* node : root->children) {
      if (node != selected) {
        delete node;
      }
    }
    root->children.clear();
    root->children.insert(selected);
  
    root = selected;

    temperature = std::pow(temperature + 1, -0.42f);

    std::cout << root->position << std::endl;
  }
}