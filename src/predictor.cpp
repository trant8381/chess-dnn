#include "predictor.h"
#include "create_state.h"
#ifdef HAS_CUDA
#include "create_state_fast.h"
#endif
#include <chrono>
#include <cstddef>
#include <cstdlib>
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

PredictorStatsSnapshot Predictor::stats() const {
  return {forwards_.load(std::memory_order_relaxed),
          positions_.load(std::memory_order_relaxed),
          minBatch_.load(std::memory_order_relaxed),
          maxBatch_.load(std::memory_order_relaxed),
          targetBatch_.load(std::memory_order_relaxed),
          settleUs_.load(std::memory_order_relaxed),
          encodeUs_.load(std::memory_order_relaxed),
          inferUs_.load(std::memory_order_relaxed)};
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
  const bool verbose = std::getenv("PREDICTOR_VERBOSE") != nullptr;

  // cuDNN autotuning + TF32 on Ampere+ for the small batch GEMMs/convs that
  // dominate self-play inference. Global; harmless to set repeatedly.
  if (device_.is_cuda()) {
    at::globalContext().setBenchmarkCuDNN(true);
    at::globalContext().setAllowTF32CuDNN(true);
    at::globalContext().setAllowTF32CuBLAS(true);
  }

  // Adaptive batching: target starts at INITIAL_BATCH and self-corrects.
  size_t target = PREDICTOR_INITIAL_BATCH;
  size_t bestTarget = target;
  double bestThpt = 0.0;
  bool probing = false;
  int quiet = 0;
  size_t windowWaves = 0;
  size_t windowPositions = 0;
  auto windowStart = std::chrono::steady_clock::now();

  while (true) {
    std::vector<std::shared_ptr<Request>> wave;
    bool shouldStop = false;

    // Settle window scales with the target: bigger targets need more time to
    // collect, otherwise the deadline (not the cap) governs wave size.
    const size_t cap = target > PREDICTOR_MIN_BATCH ? target : PREDICTOR_MIN_BATCH;
    const auto settle =
        std::chrono::microseconds(static_cast<long>(
            std::min<size_t>(20000, std::max<size_t>(500, cap * 30))));

    const auto drainStart = std::chrono::steady_clock::now();
    {
      std::unique_lock<std::mutex> lock(queueMutex_);
      queueCv_.wait(lock, [&] { return stop_ || !queue_.empty(); });

      if (!stop_ && queue_.size() < cap) {
        // Dynamic batching: give concurrent games a short window to enqueue
        // more requests so small waves coalesce into one forward pass.
        const auto deadline = std::chrono::steady_clock::now() + settle;
        while (!stop_ && queue_.size() < cap &&
               std::chrono::steady_clock::now() < deadline) {
          queueCv_.wait_until(lock, deadline);
        }
      }

      shouldStop = stop_;
      while (!queue_.empty() &&
             (shouldStop || wave.size() < cap)) {
        wave.push_back(std::move(queue_.front()));
        queue_.pop_front();
      }
    }

    const auto waveReady = std::chrono::steady_clock::now();

    if (wave.empty()) {
      if (shouldStop) {
        break;
      }
      continue; // spurious wakeup safety net
    }

    const long n = static_cast<long>(wave.size());
    const uint64_t ncast = static_cast<uint64_t>(n);
    forwards_.fetch_add(1, std::memory_order_relaxed);
    positions_.fetch_add(ncast, std::memory_order_relaxed);
    {
      uint64_t mn = minBatch_.load(std::memory_order_relaxed);
      while (mn == 0 || ncast < mn) {
        if (minBatch_.compare_exchange_weak(
                mn, ncast, std::memory_order_relaxed)) {
          break;
        }
      }
      uint64_t mx = maxBatch_.load(std::memory_order_relaxed);
      while (ncast > mx) {
        if (maxBatch_.compare_exchange_weak(
                mx, ncast, std::memory_order_relaxed)) {
          break;
        }
      }
    }

    const double settleMs =
        std::chrono::duration<double, std::milli>(waveReady - drainStart)
            .count();
    settleUs_.fetch_add(static_cast<uint64_t>(
                            std::chrono::duration_cast<std::chrono::microseconds>(
                                waveReady - drainStart)
                                .count()),
                        std::memory_order_relaxed);

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

    double encodeMs = 0.0;
    double inferMs = 0.0;
    torch::Tensor batchedInput;
    torch::Tensor policyCpu;
    torch::Tensor valueCpu;
    try {
      const auto encodeBegin = std::chrono::steady_clock::now();
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
      const auto inferBegin = std::chrono::steady_clock::now();
      encodeMs = std::chrono::duration<double, std::milli>(inferBegin -
                                                           encodeBegin)
                     .count();
      encodeUs_.fetch_add(
          static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::microseconds>(
                  inferBegin - encodeBegin)
                  .count()),
          std::memory_order_relaxed);

      Eval outputs = model_->forward(batchedInput);

      policyCpu = outputs.policy.to(torch::kCPU, torch::kFloat32)
                      .reshape({n, -1})
                      .contiguous();
      valueCpu = outputs.value.to(torch::kCPU, torch::kFloat32)
                     .reshape({n})
                     .contiguous();
      const auto inferEnd = std::chrono::steady_clock::now();
      inferMs = std::chrono::duration<double, std::milli>(inferEnd -
                                                          inferBegin)
                    .count();
      inferUs_.fetch_add(
          static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::microseconds>(
                  inferEnd - inferBegin)
                  .count()),
          std::memory_order_relaxed);
    } catch (...) {
      // Failed forward (e.g. CUDA error): fail every request of this wave so
      // the owning game threads rethrow instead of blocking forever.
      std::exception_ptr eptr = std::current_exception();
      for (const auto &request : wave) {
        request->promise.set_exception(eptr);
      }
      continue;
    }

    if (verbose) {
      std::fprintf(stderr,
                   "wave n=%ld settle=%.2fms encode=%.2fms infer=%.2fms\n",
                   n, settleMs, encodeMs, inferMs);
    }

    for (long i = 0; i < n; i++) {
      // clone(): each row owns its storage; valueCpu[i] is a 0-dim scalar,
      // policyCpu[i] a contiguous 1-D row of shape {4672}.
      wave[static_cast<size_t>(i)]->promise.set_value(
          Eval(valueCpu[i].clone(), policyCpu[i].clone()));
    }

    // --- Adaptive batch-size controller -------------------------------
    // Every ADAPT_WAVES waves (or ADAPT_SECONDS), measure achieved positions
    // per second and nudge the target: keep climbing while throughput keeps
    // improving (>2%), revert to the best target on plateau/regression, and
    // pause probing for a few windows after a revert to avoid churn.
    windowWaves += 1;
    windowPositions += static_cast<size_t>(n);
    const auto windowNow = std::chrono::steady_clock::now();
    const double windowSec =
        std::chrono::duration<double>(windowNow - windowStart).count();
    targetBatch_.store(cap, std::memory_order_relaxed);
    if (windowWaves >= static_cast<size_t>(PREDICTOR_ADAPT_WAVES) ||
        windowSec >= PREDICTOR_ADAPT_SECONDS) {
      const double thpt =
          windowSec > 0.0 ? static_cast<double>(windowPositions) / windowSec
                          : 0.0;
      const double rel =
          bestThpt > 0.0 ? thpt / bestThpt : 1.0;
      if (rel >= 1.01) {
        bestThpt = thpt;
        bestTarget = target;
      }
      const bool improve = rel >= 1.02;
      if (probing) {
        // We just measured a candidate target larger than bestTarget.
        if (improve && target < PREDICTOR_MAX_BATCH_CAP) {
          target = std::min(PREDICTOR_MAX_BATCH_CAP, target * 2);  // keep climbing
        } else {
          // Plateau or regression: settle back at the best-measured target.
          target = bestTarget;
          probing = false;
          quiet = PREDICTOR_QUIET_WINDOWS;
        }
      } else if (target < PREDICTOR_MAX_BATCH_CAP) {
        if (quiet > 0) {
          quiet -= 1;
        } else {
          target = std::min(PREDICTOR_MAX_BATCH_CAP, target * 2);  // probe upward
          probing = true;
        }
      }
      windowWaves = 0;
      windowPositions = 0;
      windowStart = windowNow;
    }
  }
}
