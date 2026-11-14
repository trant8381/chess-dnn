#pragma once

#include "dnn.h"
#include "node.h"
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <torch/torch.h>

// Adaptive batching bounds. The predictor starts at PREDICTOR_INITIAL_BATCH
// and self-corrects the target batch size while running (measuring positions
// per second over rolling windows, probing upward and reverting/halving on
// regression) so it always operates near the knee of the throughput curve.
static constexpr size_t PREDICTOR_INITIAL_BATCH = 64;
static constexpr size_t PREDICTOR_MIN_BATCH = 16;       // never go below
static constexpr size_t PREDICTOR_MAX_BATCH_CAP = 2048; // hard ceiling
static constexpr int PREDICTOR_ADAPT_WAVES = 200;       // window length (waves)
static constexpr double PREDICTOR_ADAPT_SECONDS = 2.0;  // or this many seconds
static constexpr int PREDICTOR_QUIET_WINDOWS = 5;       // rest after reverting

struct Request {
  Node *node;
  std::promise<Eval> promise;
  std::future<Eval> future;

  explicit Request(Node *_node) : node(_node), future(promise.get_future()) {}
};

// Snapshots of the predictor's running counters, mostly for performance
// investigation: how many positions actually reach each forward pass.
struct PredictorStatsSnapshot {
  uint64_t forwards = 0;  // batch forwards started
  uint64_t positions = 0; // positions evaluated in those forwards
  uint64_t minBatch = 0;  // 0 until the first forward
  uint64_t maxBatch = 0;
  uint64_t targetBatch = 0; // adaptive max-batch target in force
  uint64_t settleUs = 0;    // cumulative queue settle/wait time
  uint64_t encodeUs = 0;    // cumulative state-encode time
  uint64_t inferUs = 0;     // cumulative forward + D2H copy time
};

class Predictor {
public:
  explicit Predictor(const torch::Device &device);
  ~Predictor();

  std::shared_ptr<Request> enqueue(Node *node);

  void publishWeights(const DNN &src);

  void loadCheckpoint(const std::string &path);

  DNN &model();
  const torch::Device &device() const { return device_; }
  // Current adaptive max-batch target; collect() caps a per-epoch request
  // wave with this so one epoch maps to roughly one predictor forward.
  size_t maxBatchTarget() const {
    return targetBatch_.load(std::memory_order_relaxed);
  }
  size_t pendingRequestCount() const;
  PredictorStatsSnapshot stats() const;
  void stop();

private:
  void run();

  const torch::Device device_;
  DNN model_;
  DNN stagedModel_;
  bool weightsStaged_ = false;

  mutable std::mutex queueMutex_;
  std::condition_variable queueCv_;
  std::deque<std::shared_ptr<Request>> queue_;
  bool stop_ = false;

  std::mutex weightsMutex_;
  std::thread thread_;

  std::atomic<uint64_t> forwards_{0};
  std::atomic<uint64_t> positions_{0};
  std::atomic<uint64_t> minBatch_{0};
  std::atomic<uint64_t> maxBatch_{0};
  std::atomic<size_t> targetBatch_{PREDICTOR_INITIAL_BATCH};
  std::atomic<uint64_t> settleUs_{0};
  std::atomic<uint64_t> encodeUs_{0};
  std::atomic<uint64_t> inferUs_{0};
};
