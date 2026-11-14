#include "predictor.h"
#include "create_state.h"
#ifdef HAS_CUDA
#include "create_state_fast.h"
#endif
#include <chrono>
#include <cstddef>
#include <exception>
#include <utility>

Predictor::Predictor(const torch::Device &device) : device_(device) {
  model_->to(device_);
  model_->eval();
  stagedModel_->to(device_);
  thread_ = std::thread(&Predictor::run, this);
}

Predictor::~Predictor() {
  stop();
  if (thread_.joinable()) {
    thread_.join();
  }
}

std::shared_ptr<Request> Predictor::enqueue(Node *node) {
  auto request = std::make_shared<Request>(node);
  {
    std::lock_guard<std::mutex> lock(queueMutex_);
    queue_.push_back(request);
  }
  queueCv_.notify_one();
  return request;
}

void Predictor::publishWeights(const DNN &src) {
  std::lock_guard<std::mutex> wlock(weightsMutex_);
  copyWeights(src, stagedModel_);
  weightsStaged_ = true;
}

void Predictor::loadCheckpoint(const std::string &path) {
  torch::load(model_, path, device_);
  model_->to(device_);
  model_->eval();
}

DNN &Predictor::model() { return model_; }

size_t Predictor::pendingRequestCount() const {
  std::lock_guard<std::mutex> lock(queueMutex_);
  return queue_.size();
}

void Predictor::stop() {
  {
    std::lock_guard<std::mutex> lock(queueMutex_);
    stop_ = true;
  }
  queueCv_.notify_all();
}

void Predictor::run() {
  torch::NoGradGuard no_grad;

  while (true) {
    std::vector<std::shared_ptr<Request>> wave;
    bool shouldStop = false;

    {
      std::unique_lock<std::mutex> lock(queueMutex_);
      queueCv_.wait(lock, [&] { return stop_ || !queue_.empty(); });

      if (!stop_ && queue_.size() < static_cast<size_t>(PREDICTOR_MAX_BATCH)) {
        // Dynamic batching: give concurrent games a short window to enqueue
        // more requests so small waves coalesce into one forward pass.
        const auto deadline =
            std::chrono::steady_clock::now() + PREDICTOR_SETTLE_US;
        while (!stop_ &&
               queue_.size() < static_cast<size_t>(PREDICTOR_MAX_BATCH) &&
               std::chrono::steady_clock::now() < deadline) {
          queueCv_.wait_until(lock, deadline);
        }
      }

      shouldStop = stop_;
      while (!queue_.empty() &&
             (shouldStop ||
              wave.size() < static_cast<size_t>(PREDICTOR_MAX_BATCH))) {
        wave.push_back(std::move(queue_.front()));
        queue_.pop_front();
      }
    }

    if (wave.empty()) {
      if (shouldStop) {
        break;
      }
      continue; // spurious wakeup safety net
    }

    // Apply weights staged by publishWeights() before the forward pass, so
    // inference never races a parameter copy.
    {
      std::lock_guard<std::mutex> wlock(weightsMutex_);
      if (weightsStaged_) {
        copyWeights(stagedModel_, model_);
        weightsStaged_ = false;
      }
    }

    std::vector<Node *> nodes;
    nodes.reserve(wave.size());
    for (const auto &request : wave) {
      nodes.push_back(request->node);
    }

    torch::Tensor batchedInput;
    torch::Tensor policyCpu;
    torch::Tensor valueCpu;
    try {
#ifdef HAS_CUDA
      if (device_.is_cuda()) {
        batchedInput = createStateFast(nodes, device_);
      } else
#endif
      {
        batchedInput =
            torch::zeros({static_cast<long>(nodes.size()), INPUT_PLANES, 8, 8});
        for (size_t j = 0; j < nodes.size(); j++) {
          batchedInput[j] =
              createState(constructHistory(nodes[j]), torch::kCPU);
        }
      }

      Eval outputs = model_->forward(batchedInput);

      const long n = static_cast<long>(nodes.size());
      policyCpu = outputs.policy.to(torch::kCPU, torch::kFloat32)
                      .reshape({n, -1})
                      .contiguous();
      valueCpu = outputs.value.to(torch::kCPU, torch::kFloat32)
                     .reshape({n})
                     .contiguous();
    } catch (...) {
      // Failed forward (e.g. CUDA error): fail every request of this wave so
      // the owning game threads rethrow instead of blocking forever.
      std::exception_ptr eptr = std::current_exception();
      for (const auto &request : wave) {
        request->promise.set_exception(eptr);
      }
      continue;
    }

    const long n = static_cast<long>(nodes.size());
    for (long i = 0; i < n; i++) {
      // clone(): each row owns its storage; valueCpu[i] is a 0-dim scalar,
      // policyCpu[i] a contiguous 1-D row of shape {4672}.
      wave[static_cast<size_t>(i)]->promise.set_value(
          Eval(valueCpu[i].clone(), policyCpu[i].clone()));
    }
  }
}
