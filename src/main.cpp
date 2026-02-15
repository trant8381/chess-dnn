#include "constants.h"
#include "ctpl.h"
#include "dnn.h"
#include "mcts.h"
#include "move_gen.h"
#include <ATen/Context.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <mutex>
#include <thread>
#include <torch/cuda.h>
#include <torch/torch.h>
#include <deque>
#include <vector>
#ifdef HAS_CUDA
#include <c10/cuda/CUDAStream.h>
#endif

struct State {
  Midnight::Position position;
  float value;
};

void playGame(Node *root, DNN &model, const torch::Device device, 
  std::deque<std::vector<std::string>*>& writeQueue, std::mutex& writeQueueLock) {
  GlobalData g = GlobalData(device);
  int length = 0;

  while (true) {
    length += 1;

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

  std::vector<std::string>* fens = new std::vector<std::string>(length);

  while (root) {
    fens->push_back(root->position.fen());
    root = root->parent;
  }

  while (writeQueueLock.try_lock()) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(100));
  }
  writeQueue.push_back(fens);
  writeQueueLock.unlock();

  delete root;
}

Node *createRoot() {
  return new Node(nullptr, {}, Midnight::Position(Midnight::START_FEN));
}

int main() {
  ctpl::thread_pool pool(PARALLEL_GAMES);
  std::future<void> results[PARALLEL_GAMES];
  std::deque<std::vector<std::string>*> writeQueue;
  std::mutex writeQueueLock;

  for (size_t i = 0; i < PARALLEL_GAMES; i++) { 
    results[i] = pool.push([i, &writeQueue, &writeQueueLock](int) {
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

      playGame(root, model, device, writeQueue, writeQueueLock);
    });
  }

  while (pool.n_idle() != PARALLEL_GAMES) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  std::cout << "finished games" << std::endl;
  std::ofstream file("fens.txt");

  while (writeQueue.size() != 0) {
    std::vector<std::string>* game = writeQueue.front();
    for (std::string fen : *game) {
      file << fen << "\n";
    }
    file << "\n\n";
    writeQueue.pop_front();
    delete game;
  }

  for (int i=0; i < PARALLEL_GAMES; i++) results[i].wait();

  return 0;
}