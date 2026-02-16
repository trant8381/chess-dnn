#include "constants.h"
#include "ctpl.h"
#include "dnn.h"
#include "mcts.h"
#include "move_gen.h"
#include <ATen/Context.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/cuda.h>
#include <torch/torch.h>
#ifdef HAS_CUDA
#include <c10/cuda/CUDAStream.h>
#endif

struct State {
  Midnight::Position position;
  float value;
};

void playGame(Node *root, DNN &model, const torch::Device device, 
  std::ofstream& writeStream) {
  GlobalData g = GlobalData(device);
  int length = 0;
  
  std::vector<std::string> visitDistributions;

  while (!isTerminal(root->position)) {
    length += 1;

    float temperature = 1.0f;
    Node *selected = getNextMove(root, model, temperature, g);
    std::string visits;
    
    for (Node *node : root->children) {
      visits += std::to_string(node->visitCount) + " ";
      if (node != selected) {
        delete node;
      }
    }

    visitDistributions.push_back(visits);
    root->children.clear();
    root->children.insert(selected);

    root = selected;

    temperature = std::pow(temperature + 1, TEMPERATURE_DECAY);

    std::cout << root->position << std::endl;
  }

  float gameResult = isTerminal(root->position);
  for (int i = 0; i < length; i++) {
    root = root->parent;
    writeStream << root->position.fen() << "\t"
                << visitDistributions[length - i - 1] << "\t"
                << gameResult << "\n";
  }

  delete root;
}

Node *createRoot() {
  return new Node(nullptr, {}, Midnight::Position(Midnight::START_FEN));
}

int main() {
  ctpl::thread_pool pool(PARALLEL_GAMES);
  std::future<void> results[PARALLEL_GAMES];

  std::filesystem::create_directory("train");

  for (size_t i = 0; i < PARALLEL_GAMES; i++) {
    results[i] = pool.push([i](int) {
      std::ofstream writeStream("train/" + std::to_string(i) + ".tsv");

      Node *root = createRoot();
      torch::Device device = torch::kCPU;
      #ifdef HAS_CUDA 
      at::cuda::CUDAStream myStream = at::cuda::getStreamFromPool();
      at::cuda::setCurrentCUDAStream(myStream);
      device = torch::Device(torch::kCUDA, i % torch::getNumGPUs());
      #endif

      DNN model = DNN();
      torch::NoGradGuard no_grad;
      model->to(device);

      playGame(root, model, device, writeStream);
    });
  }

  for (int i=0; i < PARALLEL_GAMES; i++) results[i].wait();

  return 0;
}