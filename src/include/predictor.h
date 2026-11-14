#pragma once

#include "dnn.h"
#include "node.h"
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <torch/torch.h>

static constexpr int PREDICTOR_MAX_BATCH = 64;
static constexpr std::chrono::microseconds PREDICTOR_SETTLE_US{2000};

struct Request {
  Node *node;
  std::promise<Eval> promise;
  std::future<Eval> future;

  explicit Request(Node *_node) : node(_node), future(promise.get_future()) {}
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
  size_t pendingRequestCount() const;
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
};
