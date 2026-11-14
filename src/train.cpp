#include "constants.h"
#include "ctpl.h"
#include "dnn.h"
#include "mcts.h"
#include "move_gen.h"
#include "predictor.h"
#include <memory>
#include <ATen/Context.h>
#include <atomic>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <chrono>
#include <cmath>
#include <ctime>
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

// Frequently-tuned knobs, overridable on the command line (see main()).
// Model-size and structural parameters stay compile-time in constants.h.
struct Config {
  int totalGames = TOTAL_GAMES;
  size_t replayCapacity = REPLAY_CAPACITY;
  size_t trainMinSamples = TRAIN_MIN_SAMPLES;
  int64_t trainBatchSize = TRAIN_BATCH_SIZE;
  int parallelGames = PARALLEL_GAMES;
  int simulations = SIMULATIONS; // per-move MCTS search budget
  double learningRate = LEARNING_RATE;
};

struct State {
  Midnight::Position position;
  float value;
};

class ReplayBuffer {
public:
  explicit ReplayBuffer(size_t capacity) : capacity_(capacity) {}

  void add(std::vector<Sample> &&samples) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (Sample &sample : samples) {
      data_.push_back(std::move(sample));
      if (data_.size() > capacity_) {
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
  const size_t capacity_;
};

struct SharedWeights {
  std::mutex mutex;
  DNN model;
  std::atomic<int> version{0};
};

// Shared counters for the training log; self-play workers increment,
// trainLoop samples them each step and on exit.
struct TrainingStats {
  std::atomic<int64_t> gamesCompleted{0};  // games that reached a terminal
  std::atomic<int64_t> gamesDrawn{0};      // terminal value 0 (stalemate,
                                           // fifty-move, threefold,
                                           // insufficient material)
  std::atomic<int64_t> gamesAborted{0};    // ended with no terminal (no
                                           // selectable move)
  std::atomic<int64_t> pliesPlayed{0};     // plies of completed games
};

// Log per-predictor wave statistics (batch sizes actually reaching the
// model, settle/encode/infer time, positions per second).
static void reportPredictorStats(
    std::ostream &out,
    const std::vector<std::unique_ptr<Predictor>> &predictors,
    double elapsed) {
  for (size_t i = 0; i < predictors.size(); i++) {
    const PredictorStatsSnapshot ps = predictors[i]->stats();
    const uint64_t forwards = ps.forwards;
    const uint64_t positions = ps.positions;
    const uint64_t avgBatch = forwards ? positions / forwards : 0;
    const double waveMs =
        forwards
            ? static_cast<double>(ps.settleUs + ps.encodeUs + ps.inferUs) /
                  (1000.0 * static_cast<double>(forwards))
            : 0.0;
    out << "# predictor[" << i << "] forwards=" << forwards
        << " positions=" << positions << " avg_batch=" << avgBatch
        << " min_batch=" << (forwards ? ps.minBatch : 0)
        << " max_batch=" << ps.maxBatch
        << " target_batch=" << ps.targetBatch
        << " wave_ms=" << waveMs
        << " (settle="
        << (forwards ? ps.settleUs / (1000.0 * static_cast<double>(forwards))
                     : 0.0)
        << " encode="
        << (forwards ? ps.encodeUs / (1000.0 * static_cast<double>(forwards))
                     : 0.0)
        << " infer="
        << (forwards ? ps.inferUs / (1000.0 * static_cast<double>(forwards))
                     : 0.0)
        << ") pos_per_s="
        << (elapsed > 0.0 ? positions / elapsed : 0.0) << '\n';
  }
}

static void publishWeights(DNN &trained, SharedWeights &shared) {
  std::lock_guard<std::mutex> lock(shared.mutex);
  copyWeights(trained, shared.model);
  shared.version.fetch_add(1);
}

static void syncWeights(SharedWeights &shared, Predictor &predictor,
                        int &localVersion) {
  if (shared.version.load() == localVersion) {
    return;
  }
  std::lock_guard<std::mutex> lock(shared.mutex);
  predictor.publishWeights(shared.model);
  localVersion = shared.version.load();
}

static void trainLoop(ReplayBuffer &buffer, SharedWeights &shared,
                      const torch::Device device, std::atomic<bool> &stop,
                      TrainingStats &stats,
                      const std::vector<std::unique_ptr<Predictor>>
                          &predictors,
                      const Config &cfg) {
  DNN model;
  model->to(device);
  model->train();
  copyWeights(shared.model, model);

  torch::optim::SGD optimizer(
      model->parameters(),
      torch::optim::SGDOptions(cfg.learningRate).momentum(0.9).weight_decay(1e-4));

  std::mt19937_64 rng{std::random_device{}()};
  int step = 0;

  // CSV training log, one row per completed train step plus event markers
  // and a final run summary. Rewritten at the start of each run.
  std::ofstream logFile("training.log", std::ios::out | std::ios::trunc);
  if (!logFile) {
    std::cerr << "warning: could not open training.log" << std::endl;
  }
  const auto runStart = std::chrono::steady_clock::now();
  if (logFile) {
    logFile << "# chess-dnn training run " << std::time(nullptr)
            << " total_games=" << cfg.totalGames
            << " replay_capacity=" << cfg.replayCapacity
            << " batch=" << cfg.trainBatchSize
            << " min_samples=" << cfg.trainMinSamples
            << " lr=" << cfg.learningRate
            << " simulations=" << cfg.simulations
            << " parallel_games=" << cfg.parallelGames
            << " device=" << device.str() << '\n';
    logFile << "step,policy_loss,value_loss,total_loss,grad_norm,"
               "games_completed,games_aborted,draw_rate,"
               "avg_plies_per_game,replay_samples,replay_fill_pct,"
               "elapsed_s,games_per_s\n";
    logFile.flush();
  }
  bool reportedReady = false;

  while (!stop.load()) {
    if (buffer.size() < cfg.trainMinSamples) {
      if (!reportedReady && logFile) {
        logFile << "# replay buffer filling: " << buffer.size() << "/"
                << cfg.trainMinSamples << " samples (train waits below this)"
                << '\n';
        logFile.flush();
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      continue;
    }
    if (!reportedReady) {
      reportedReady = true;
      if (logFile) {
        logFile << "# replay buffer reached min size: " << buffer.size()
                << " samples; training starts" << '\n';
        logFile.flush();
      }
    }

    std::vector<Sample> batch = buffer.sample(cfg.trainBatchSize, rng);
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

    if (logFile) {
      const int64_t gc = stats.gamesCompleted.load();
      const double drawRate =
          gc > 0 ? static_cast<double>(stats.gamesDrawn.load()) / gc : 0.0;
      const double avgPlies =
          gc > 0 ? static_cast<double>(stats.pliesPlayed.load()) / gc : 0.0;
      const double elapsed =
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        runStart)
              .count();
      logFile << step << ',' << policyLoss.item<float>() << ','
              << valueLoss.item<float>() << ',' << loss.item<float>() << ','
              << gradNorm << ',' << gc << ','
              << stats.gamesAborted.load() << ',' << drawRate << ','
              << avgPlies << ',' << buffer.size() << ','
              << (100.0 * static_cast<double>(buffer.size()) /
                  static_cast<double>(cfg.replayCapacity))
              << ',' << elapsed << ',' << (elapsed > 0 ? gc / elapsed : 0.0)
              << '\n';
      logFile.flush();
    }

    if (step % TRAIN_PUBLISH_EVERY == 0) {
      publishWeights(model, shared);
      std::cout << "train step " << step
                << " policy loss " << policyLoss.item<float>()
                << " value loss " << valueLoss.item<float>() << std::endl;
      const double elapsed =
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        runStart)
              .count();
      reportPredictorStats(std::cout, predictors, elapsed);
      if (logFile) {
        reportPredictorStats(logFile, predictors, elapsed);
      }
    }
    if (step % TRAIN_CHECKPOINT_EVERY == 0) {
      torch::save(model, "checkpoint.pt");
      if (logFile) {
        logFile << "# checkpoint.pt saved at step " << step << '\n';
        logFile.flush();
      }
    }
    std::cout << step << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(10));
  }

  if (logFile) {
    const int64_t gc = stats.gamesCompleted.load();
    const double drawRate =
        gc > 0 ? static_cast<double>(stats.gamesDrawn.load()) / gc : 0.0;
    const double avgPlies =
        gc > 0 ? static_cast<double>(stats.pliesPlayed.load()) / gc : 0.0;
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      runStart)
            .count();
    logFile << "# run end: train_steps=" << step
            << " games_completed=" << gc
            << " drawn=" << stats.gamesDrawn.load()
            << " aborted=" << stats.gamesAborted.load()
            << " draw_rate=" << drawRate
            << " avg_plies_per_game=" << avgPlies
            << " elapsed_s=" << elapsed
            << " games_per_s=" << (elapsed > 0 ? gc / elapsed : 0.0) << '\n';
    reportPredictorStats(logFile, predictors, elapsed);
    logFile.flush();
  }
  {
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      runStart)
            .count();
    reportPredictorStats(std::cout, predictors, elapsed);
  }

  publishWeights(model, shared);
  torch::save(model, "checkpoint.pt");
}

static void playGame(Node *root, Predictor &predictor,
                     const torch::Device device, ReplayBuffer &buffer,
                     TrainingStats &stats, int simulations) {
  GlobalData g = GlobalData(device, &predictor);
  g.simulations = simulations;

  float temperature = 1.0f;
  std::vector<Sample> samples; // one per ply, in game order

  while (!isTerminal(root->position)) {
    Node *selected = getNextMove(root, temperature, g);
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
    stats.gamesCompleted.fetch_add(1);
    stats.pliesPlayed.fetch_add(static_cast<int64_t>(samples.size()));
    if (z == 0.0f) {
      stats.gamesDrawn.fetch_add(1);
    }
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
  } else {
    stats.gamesAborted.fetch_add(1);
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
                           std::atomic<int> &gamesStarted,
                           Predictor *predictor, TrainingStats &stats,
                           int simulations, int totalGames) {
  torch::Device device = torch::kCPU;
  if (torch::cuda::is_available()) {
    device = torch::Device(torch::kCUDA, id % torch::getNumGPUs());
  }

  int version = -1;
  while (gamesStarted.fetch_add(1) < totalGames) {
    syncWeights(shared, *predictor, version);
    playGame(createRoot(), *predictor, device, buffer, stats, simulations);
  }
}

// --key value ... parser: returns the value for key, or defaultValue.
static int cliInt(int argc, char **argv, const char *key, int defaultValue) {
  for (int i = 1; i < argc - 1; i++) {
    if (std::string(argv[i]) == key) {
      return std::atoi(argv[i + 1]);
    }
  }
  return defaultValue;
}

int main(int argc, char **argv) {
  Config cfg;
  cfg.totalGames =
      cliInt(argc, argv, "--total-games", cfg.totalGames);
  cfg.parallelGames =
      cliInt(argc, argv, "--par-games", cfg.parallelGames);
  cfg.simulations =
      cliInt(argc, argv, "--simulations", cfg.simulations);
  cfg.trainMinSamples = static_cast<size_t>(
      cliInt(argc, argv, "--min-samples", static_cast<int>(cfg.trainMinSamples)));
  cfg.trainBatchSize = static_cast<int64_t>(
      cliInt(argc, argv, "--train-batch", static_cast<int>(cfg.trainBatchSize)));
  cfg.replayCapacity = static_cast<size_t>(
      cliInt(argc, argv, "--replay-capacity", static_cast<int>(cfg.replayCapacity)));
  cfg.learningRate = static_cast<double>(
      cliInt(argc, argv, "--lr", static_cast<int>(cfg.learningRate * 1e3))) /
                     1e3;

  if (argc > 1 && std::string(argv[1]) == "--help") {
    std::cout << "usage: train [--total-games N] [--par-games N]"
                 " [--simulations N] [--min-samples N] [--train-batch N]"
                 " [--replay-capacity N] [--lr 1e-3]\n";
    return 0;
  }

  ReplayBuffer buffer(cfg.replayCapacity);
  SharedWeights shared;
  // torch::load(shared.model, "checkpoint.pt");

  std::atomic<bool> stopTraining{false};
  std::atomic<int> gamesStarted{0};
  TrainingStats stats;

  torch::Device trainDevice = torch::kCPU;
  if (torch::cuda::is_available()) {
    trainDevice = torch::Device(torch::kCUDA, 0);
    at::globalContext().setBenchmarkCuDNN(true);
    at::globalContext().setAllowTF32CuDNN(true);
    at::globalContext().setAllowTF32CuBLAS(true);
  }

  // One central predictor per GPU (plus a CPU fallback); game workers no
  // longer own model copies.
  std::vector<std::unique_ptr<Predictor>> predictors;
  if (torch::cuda::is_available()) {
    const int64_t numGPUs = torch::getNumGPUs();
    for (int64_t i = 0; i < numGPUs; i++) {
      predictors.push_back(
          std::make_unique<Predictor>(torch::Device(torch::kCUDA, i)));
    }
  } else {
    predictors.push_back(std::make_unique<Predictor>(torch::kCPU));
  }

  std::future<void> trainer = std::async(std::launch::async, [&] {
    trainLoop(buffer, shared, trainDevice, stopTraining, stats, predictors,
              cfg);
  });

  std::exception_ptr error;
  try {
    ctpl::thread_pool pool(cfg.parallelGames);
    std::vector<std::future<void>> games;
    for (int i = 0; i < cfg.parallelGames; i++) {
      games.push_back(pool.push([&, i](int) {
        selfPlayWorker(static_cast<size_t>(i), buffer, shared, gamesStarted,
                       predictors[i % predictors.size()].get(), stats,
                       cfg.simulations, cfg.totalGames);
      }));
    }

    for (std::future<void> &game : games) {
      game.get();
    }
  } catch (...) {
    error = std::current_exception();
  }

  for (auto &predictor : predictors) {
    predictor->stop();  // drains any remaining queued requests
  }

  stopTraining = true;
  trainer.get();

  if (error) {
    std::rethrow_exception(error);
  }
  return 0;
}
