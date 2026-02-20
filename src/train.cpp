#include "constants.h"
#include "create_state.h"
#include "ctpl.h"
#include "dnn.h"
#include "mcts.h"
#include "move_gen.h"
#include "playBuffer.h"
#include <ATen/Context.h>
#include <ATen/core/jit_type.h>
#include <ATen/ops/_sample_dirichlet.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/cuda.h>
#include <torch/nn/functional/activation.h>
#include <torch/nn/functional/loss.h>
#include <torch/nn/modules/loss.h>
#include <torch/nn/options/activation.h>
#include <torch/optim/sgd.h>
#include <torch/serialize.h>
#include <torch/torch.h>
#ifdef HAS_CUDA
#include "create_state_fast.h"
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
  }

  float gameResult = terminalValue(root->position);
  while (buffer.lock.try_lock()) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }

  for (int i = 0; i < length; i++) {
    root = root->parent;
    GameStats gameStats = GameStats(
        root->position.fen(), visitDistributions[length - i - 1], gameResult);
    buffer.insert(std::move(gameStats));
  }

  buffer.lock.unlock();
  delete root;
}

Node *createRoot() {
  return new Node(nullptr, {}, Midnight::Position(Midnight::START_FEN));
}

torch::Tensor computeLoss(const torch::Tensor &policyMask,
                          const torch::Tensor &valueTargets,
                          const torch::Tensor &policyTargets,
                          const Eval &eval) {
  torch::Tensor masked = eval.policy.masked_fill(~policyMask, 0);
  torch::Tensor logProbs = torch::nn::functional::log_softmax(
      masked, torch::nn::functional::LogSoftmaxFuncOptions(1));
  torch::Tensor policyLoss = torch::sum(-(policyTargets * logProbs), 1).mean();
  torch::Tensor valueLoss =
      torch::nn::functional::mse_loss(eval.value.squeeze(-1), valueTargets);

  return policyLoss + valueLoss;
}

int main() {
  PlayBuffer buffer;
  DNN _model = DNN();
  torch::save(_model, "model.pt");
  std::mutex fileLock;

  std::thread gameThread([&buffer, &fileLock]() {
    std::cout << "gameThread started" << std::endl;
    ctpl::thread_pool pool(PARALLEL_GAMES);
    std::atomic<int32_t> games;
    while (true) {
      int idleThreads = pool.n_idle();
      if (idleThreads) {
        for (int i = 0; i < idleThreads; i++) {
          pool.push([i, &buffer, &fileLock, &games](int) {
            Node *root = createRoot();
            torch::Device device = torch::kCPU;
#ifdef HAS_CUDA
            at::cuda::CUDAStream myStream = at::cuda::getStreamFromPool();
            at::cuda::setCurrentCUDAStream(myStream);
            device = torch::Device(torch::kCUDA, i % torch::getNumGPUs());
#endif

            DNN model = DNN();
            while (fileLock.try_lock()) {
              std::this_thread::sleep_for(std::chrono::nanoseconds(100));
            }
            torch::load(model, "model.pt");
            fileLock.unlock();

            torch::NoGradGuard no_grad;
            model->to(device);

            playGame(root, model, device, buffer);
            ++games;
          });
        }
      } else {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        std::cout << games << " games played" << std::endl;
      }
    }
  });

  std::thread trainThread([&buffer, &fileLock]() {
    torch::Device device = torch::kCPU;
    DNN model = DNN();
    torch::load(model, "model.pt");
    torch::optim::SGD optimizer = torch::optim::SGD(
        model->parameters(),
        torch::optim::SGDOptions(0.1).momentum(0.9).weight_decay(1e-4).nesterov(
            false));

    model->to(device);
    model->train(true);

    std::cout << "trainThread started" << std::endl;
    uint64_t currSize = buffer.totalSize();

    while (true) {
      if (buffer.totalSize() - currSize >= TRAIN_THRESHOLD) {
        currSize = buffer.totalSize();
        std::vector<GameStats> samp = buffer.sample();
        std::vector<Node *> positions;
        torch::Tensor policyTargets =
            torch::zeros({static_cast<long>(samp.size()), 4672}).to(device);
        torch::Tensor mask =
            torch::zeros({static_cast<long>(samp.size()), 4672},
                         torch::TensorOptions().dtype(torch::kBool))
                .to(device);
        torch::Tensor valueTargets =
            torch::zeros({static_cast<long>(samp.size())}).to(device);

        for (size_t i = 0; i < samp.size(); i++) {
          GameStats gameStats = samp[i];
          Midnight::Position position = Midnight::Position(gameStats.fen);
          positions.push_back(new Node(nullptr, {}, position));
          std::vector<Move> movelist = createMovelistVec(position);

          uint32_t sum = 0;
          for (size_t j = 0; j < movelist.size(); j++) {
            mask[i][policyIndex(position, movelist[j])] = 1;
            policyTargets[i][policyIndex(position, movelist[j])] =
                gameStats.distribution[j];
            sum += gameStats.distribution[j];
          }

          policyTargets[i] /= sum;
          valueTargets[i] = gameStats.result;
        }

        torch::Tensor state;

        if (torch::cuda::is_available()) {
#if HAS_CUDA
          state = createStateFast(positions, device);
#endif
        } else {
          state = torch::zeros(
              {static_cast<long>(positions.size()), INPUT_PLANES, 8, 8},
              torch::TensorOptions(device));
          for (size_t i = 0; i < positions.size(); i++) {
            state[i] = createState(constructHistory(positions[i]), device);
          }
        }

        for (Node *node : positions) {
          delete node;
        }

        state.set_requires_grad(true);

        Eval eval = model->forward(state);
        torch::Tensor loss =
            computeLoss(mask, valueTargets, policyTargets, eval);

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        std::cout << "train loss: " << loss << std::endl;

        while (fileLock.try_lock()) {
          std::this_thread::sleep_for(std::chrono::nanoseconds(100));
        }
        torch::save(model, "model.pt");
        fileLock.unlock();
      } else {
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }
    }
  });

  gameThread.join();
  trainThread.join();

  return 0;
}