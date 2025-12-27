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
  Node root = Node({}, Midnight::Position(Midnight::START_FEN), 0);

  while (true) {
    if (isTerminal(root.position)) {
      break;
    }
    float temperature = 1.0f;
    root = playMove(&root, model, device, temperature);
    temperature = std::pow(temperature + 1, -0.42f);

    std::cout << root.position << std::endl;
  }
}