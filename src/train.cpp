#include "constants.h"
#include "ctpl.h"
#include "dnn.h"
#include "mcts.h"
#include "move_gen.h"
#include <ATen/Context.h>
#include <atomic>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <deque>
#include <exception>
#include <future>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <torch/cuda.h>
#include <torch/torch.h>
#include <utility>
#include <vector>
#include <fstream>

static constexpr int TOTAL_GAMES = 100000;         // self-play games to run
static constexpr size_t REPLAY_CAPACITY = 500000;  // positions kept
static constexpr size_t TRAIN_MIN_SAMPLES = 100000;  // wait for this many first
static constexpr int64_t TRAIN_BATCH_SIZE = 256;
static constexpr int TRAIN_PUBLISH_EVERY = 100;    // steps between publishes
static constexpr int TRAIN_CHECKPOINT_EVERY = 1000;
static constexpr double LEARNING_RATE = 0.001;

struct State {
  Midnight::Position position;
  float value;
};

class ReplayBuffer {
public:
  void add(std::vector<Sample> &&samples) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (Sample &sample : samples) {
      data_.push_back(std::move(sample));
      if (data_.size() > REPLAY_CAPACITY) {
        data_.pop_front();
      }
    }
  }

  size_t size() {
    std::lock_guard<std::mutex> lock(mutex_);
    return data_.size();
  }

  std::vector<Sample> sample(size_t n, std::mt19937_64 &rng) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<Sample> out;
    if (data_.empty()) {
      return out;
    }
    out.reserve(n);
    std::uniform_int_distribution<size_t> pick(0, data_.size() - 1);
    for (size_t i = 0; i < n; i++) {
      out.push_back(data_[pick(rng)]);
    }
    return out;
  }

private:
  std::mutex mutex_;
  std::deque<Sample> data_;
};

struct SharedWeights {
  std::mutex mutex;
  DNN model;
  std::atomic<int> version{0};
};

static void copyWeights(DNN &src, DNN &dst) {
  torch::NoGradGuard no_grad;

  auto srcParams = src->named_parameters();
  auto dstParams = dst->named_parameters();
  for (auto &item : dstParams) {
    item.value().copy_(srcParams[item.key()]);
  }

  auto srcBuffers = src->named_buffers();
  auto dstBuffers = dst->named_buffers();
  for (auto &item : dstBuffers) {
    item.value().copy_(srcBuffers[item.key()]);
  }
}

static void publishWeights(DNN &trained, SharedWeights &shared) {
  std::lock_guard<std::mutex> lock(shared.mutex);
  copyWeights(trained, shared.model);
  shared.version.fetch_add(1);
}

static void syncWeights(SharedWeights &shared, DNN &local, int &localVersion) {
  if (shared.version.load() == localVersion) {
    return;
  }
  std::lock_guard<std::mutex> lock(shared.mutex);
  copyWeights(shared.model, local);
  localVersion = shared.version.load();
}

static void trainLoop(ReplayBuffer &buffer, SharedWeights &shared,
                      const torch::Device device, std::atomic<bool> &stop) {
  DNN model;
  model->to(device);
  model->train();
  copyWeights(shared.model, model);

  torch::optim::SGD optimizer(
      model->parameters(),
      torch::optim::SGDOptions(LEARNING_RATE).momentum(0.9).weight_decay(1e-4));

  std::mt19937_64 rng{std::random_device{}()};
  int step = 0;

  while (!stop.load()) {
    if (buffer.size() < TRAIN_MIN_SAMPLES) {
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      continue;
    }

    std::vector<Sample> batch = buffer.sample(TRAIN_BATCH_SIZE, rng);
    const int64_t n = static_cast<int64_t>(batch.size());
    if (n == 0) {
      continue;
    }

    std::vector<torch::Tensor> inputs;
    inputs.reserve(n);
    torch::Tensor values = torch::empty({n});
    auto valueAcc = values.accessor<float, 1>();
    for (int64_t i = 0; i < n; i++) {
      inputs.push_back(batch[i].input);
      valueAcc[i] = batch[i].value;
    }

    Eval out = model->forward(torch::stack(inputs).to(device));
    torch::Tensor policy = out.policy.reshape({n, -1});
    torch::Tensor value = out.value.reshape({n});

    // Dense policy target built from the sparse visit shares.
    torch::Tensor target = torch::zeros({n, policy.size(1)});
    auto targetAcc = target.accessor<float, 2>();
    for (int64_t i = 0; i < n; i++) {
      for (const std::pair<int, float> &entry : batch[i].policy) {
        targetAcc[i][entry.first] = entry.second;
      }
    }

    torch::Tensor logProb = POLICY_IS_LOGITS
                                ? torch::log_softmax(policy, 1)
                                : torch::log(policy.clamp_min(1e-8));
    torch::Tensor policyLoss = -(target.to(device) * logProb).sum(1).mean();
    torch::Tensor valueLoss = torch::mse_loss(value, values.to(device));
    torch::Tensor loss = policyLoss + valueLoss;

    if (!std::isfinite(loss.item<float>())) {
      std::cerr << "skipping train step: non-finite loss" << std::endl;
      continue;
    }

    optimizer.zero_grad();
    loss.backward();
    const double gradNorm =
        torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
    if (!std::isfinite(gradNorm)) {
      std::cerr << "skipping train step: non-finite gradient" << std::endl;
      optimizer.zero_grad();
      continue;
    }
    optimizer.step();
    step++;

    if (step % TRAIN_PUBLISH_EVERY == 0) {
      publishWeights(model, shared);
      std::cout << "train step " << step
                << " policy loss " << policyLoss.item<float>()
                << " value loss " << valueLoss.item<float>() << std::endl;
    }
    if (step % TRAIN_CHECKPOINT_EVERY == 0) {
      torch::save(model, "checkpoint.pt");
    }
    std::cout << step << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(10));
  }

  publishWeights(model, shared);
  torch::save(model, "checkpoint.pt");
}

static void playGame(Node *root, DNN &model, const torch::Device device,
                     ReplayBuffer &buffer) {
  GlobalData g = GlobalData(device);

  float temperature = 1.0f;
  std::vector<Sample> samples; // one per ply, in game order

  while (!isTerminal(root->position)) {
    Node *selected = getNextMove(root, model, temperature, g);
    if (selected == nullptr) {
      break; // nothing to play
    }

    samples.push_back(makeSample(root));

    // Keep only the chosen subtree.
    for (Node *node : root->children) {
      if (node != selected) {
        delete node;
      }
    }
    root->children.clear();
    root->children.insert(selected);

    root = selected;

    // Assumes TEMPERATURE_DECAY is a factor below 1 (e.g. 0.97).
    temperature *= TEMPERATURE_DECAY;
  }

  if (isTerminal(root->position)) {
    const float z = terminalValue(root->position);
    const size_t plies = samples.size();

    std::vector<Sample> usable;
    usable.reserve(plies);
    for (size_t t = 0; t < plies; t++) {
      Sample &sample = samples[t];
      if (sample.policy.empty()) {
        continue;
      }
      sample.value = ((plies - t) % 2 == 0) ? z : -z;
      usable.push_back(std::move(sample));
    }
    buffer.add(std::move(usable));
  }

  Node *first = root;
  for (Node *node = root; node; node = node->parent) {
    first = node;
  }

  delete first;
}

static Node *createRoot() {
  return new Node(nullptr, {}, Midnight::Position(Midnight::START_FEN));
}

static void selfPlayWorker(size_t id, ReplayBuffer &buffer,
                           SharedWeights &shared,
                           std::atomic<int> &gamesStarted) {
  torch::Device device = torch::kCPU;
  if (torch::cuda::is_available()) {
    device = torch::Device(torch::kCUDA, id % torch::getNumGPUs());
  }

  DNN model;
  model->to(device);
  model->eval();
  torch::NoGradGuard no_grad;

  int version = -1;
  while (gamesStarted.fetch_add(1) < TOTAL_GAMES) {
    syncWeights(shared, model, version);
    playGame(createRoot(), model, device, buffer);
  }
}

int main() {
  ReplayBuffer buffer;
  SharedWeights shared;
  // torch::load(shared.model, "checkpoint.pt");

  std::atomic<bool> stopTraining{false};
  std::atomic<int> gamesStarted{0};

  torch::Device trainDevice = torch::kCPU;
  if (torch::cuda::is_available()) {
    trainDevice = torch::Device(torch::kCUDA, 0);
  }

  std::future<void> trainer = std::async(std::launch::async, [&] {
    trainLoop(buffer, shared, trainDevice, stopTraining);
  });

  std::exception_ptr error;
  try {
    ctpl::thread_pool pool(PARALLEL_GAMES);
    std::vector<std::future<void>> games;
    for (size_t i = 0; i < PARALLEL_GAMES; i++) {
      games.push_back(pool.push([&, i](int) {
        selfPlayWorker(i, buffer, shared, gamesStarted);
      }));
    }

    for (std::future<void> &game : games) {
      game.get();
    }
  } catch (...) {
    error = std::current_exception();
  }

  stopTraining = true;
  trainer.get();

  if (error) {
    std::rethrow_exception(error);
  }
  return 0;
}
