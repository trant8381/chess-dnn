#include "concurrent_queue.h"
#include "constants.h"
#include "ctpl.h"
#include "dnn.h"
#include "evaluate.h"
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

void playGame(Node *root, DNN &model, GlobalData &g) {
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
  Node *root = new Node(nullptr, {}, Midnight::Position(Midnight::START_FEN));
  return root;
}


int main() {
  ctpl::thread_pool pool(PARALLEL_GAMES);
  std::array<GlobalData*, PARALLEL_GAMES> globalData;
  moodycamel::ConcurrentQueue<Node*> q;

  for (size_t i = 0; i < PARALLEL_GAMES; i++) {
    pool.push([i, &q, &globalData](int) {
      Node *root = createRoot();
      
      torch::Device device = torch::kCPU;
      if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA, i % torch::getNumGPUs());
      }

      GlobalData g = GlobalData(device, &q);
      globalData[i] = &g;

      DNN model = DNN();
      torch::NoGradGuard no_grad;
      model->to(device);
      root->threadIndex = i;
      playGame(root, model, g);
    });
  }

  #if HAS_CUDA
  DNN model = DNN();
  std::thread evaluateThread = std::thread([&](){
    while (pool.n_idle() != PARALLEL_GAMES) {
      evaluate(q, model, globalData);
      std::this_thread::sleep_for(std::chrono::nanoseconds(10000));
    }
  });
  evaluateThread.join();
  #endif

  return 0;
}