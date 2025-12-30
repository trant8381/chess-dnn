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

  Node *root = new Node({}, Midnight::Position(Midnight::START_FEN), 0);

  // for (Midnight::Move move : createMovelistVec(root->position)) {
  //   root->position.play<Midnight::WHITE>(move);
  //   Node node = Node({}, root->position, 0);
  //   root->children.push_back(&node);
  //   root->position.undo<Midnight::WHITE>(move);
  // }

  while (true) {
    if (isTerminal(root->position)) {
      break;
    }
    float temperature = 1.0f;
    Node* selected = playMove(root, model, device, temperature);
    for (Node* node : root->children) {
      if (node != selected) {
        delete node;
      }
    }
    root->children.clear();
    root->children.push_back(selected);
  
    root = selected;

    temperature = std::pow(temperature + 1, -0.42f);

    std::cout << root->position << std::endl;
  }
}