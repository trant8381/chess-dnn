#include "constants.h"
#include "ctpl.h"
#include "dnn.h"
#include "mcts.h"
#include "move_gen.h"
#include "playBuffer.h"
#include <ATen/Context.h>
#include <ATen/ops/_sample_dirichlet.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/cuda.h>
#include <torch/serialize.h>
#include <torch/torch.h>
#ifdef HAS_CUDA
#include <c10/cuda/CUDAStream.h>
#endif

struct State {
  Midnight::Position position;
  float value;
};

void playGame(Node *root, DNN &model, const torch::Device device,
              PlayBuffer &buffer) {
  GlobalData g = GlobalData(device);
  int length = 0;

  std::vector<std::vector<int32_t>> visitDistributions;

  while (!isTerminal(root->position)) {
    length += 1;

    float temperature = 1.0f;
    Node *selected = getNextMove(root, model, temperature, g);
    std::vector<int32_t> visits;

    for (Node *node : root->children) {
      visits.push_back(node->visitCount);
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

  float gameResult = terminalValue(root->position);
  while (buffer.lock.try_lock()) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }

  for (int i = 0; i < length; i++) {
    root = root->parent;
    GameStats gameStats = GameStats(root->position.fen(),
                            visitDistributions[length - i - 1],
                            gameResult);
    buffer.insert(std::move(gameStats));
  }

  buffer.lock.unlock();
  delete root;
}

Node *createRoot() {
  return new Node(nullptr, {}, Midnight::Position(Midnight::START_FEN));
}

int main() {
  ctpl::thread_pool pool(PARALLEL_GAMES);
  std::future<void> results[PARALLEL_GAMES];

  PlayBuffer buffer;
  DNN _model = DNN();
  torch::save(_model, "model.pt");
  for (size_t i = 0; i < PARALLEL_GAMES; i++) {
    results[i] = pool.push([i, &buffer](int) {
      Node *root = createRoot();
      torch::Device device = torch::kCPU;
#ifdef HAS_CUDA
      at::cuda::CUDAStream myStream = at::cuda::getStreamFromPool();
      at::cuda::setCurrentCUDAStream(myStream);
      device = torch::Device(torch::kCUDA, i % torch::getNumGPUs());
#endif

      DNN model = DNN();
      torch::load(model, "model.pt");
      torch::NoGradGuard no_grad;
      model->to(device);

      playGame(root, model, device, buffer);
    });
  }

  for (int i = 0; i < PARALLEL_GAMES; i++) {
    results[i].wait();
  }
  
  return 0;
}