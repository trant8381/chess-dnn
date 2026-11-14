// Predictor queue: eval delivery, tensor shapes, correctness against a
// direct forward on the same weights, weight publishing, thread contention
// and stop() drain semantics. Runs on CPU and GPU (argv: cpu / gpu).
#include "test_util.h"
#include <future>
#include <mutex>
#include <thread>

int main(int argc, char **argv) {
  torch::Device dev = testutil::pickDevice(argc, argv);
  printf("device: %s\n", dev.str().c_str());

  const std::string fens[] = {
      START_FEN,
      "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
      "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
      "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
      "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 6 4",
  };
  std::vector<Node *> nodes;
  for (const std::string &f : fens) nodes.push_back(testutil::makeChain(f, 4));
  const size_t nNodes = nodes.size();

  // Reference: CPU clone with the same weights (eval mode is deterministic).
  Predictor predictor(dev);
  DNN refModel;
  copyWeights(predictor.model(), refModel);
  refModel->to(torch::kCPU);
  refModel->eval();

  auto refEval = [&](Node *n) {
    torch::Tensor in =
        createState(constructHistory(n), torch::kCPU).unsqueeze(0);
    return refModel->forward(in);
  };

  // 1) single-wave round trip: every request resolves to a CPU Eval
  std::vector<std::shared_ptr<Request>> reqs;
  for (Node *n : nodes) reqs.push_back(predictor.enqueue(n));
  REQUIRE(predictor.pendingRequestCount() <= nNodes);
  std::vector<Eval> evals;
  for (auto &r : reqs) evals.push_back(r->future.get());
  REQUIRE_EQ(evals.size(), nNodes);
  for (size_t i = 0; i < evals.size(); i++) {
    REQUIRE(evals[i].value.device() == torch::kCPU);
    REQUIRE_EQ(evals[i].value.numel(), 1L);
    REQUIRE(evals[i].policy.device() == torch::kCPU);
    REQUIRE(evals[i].policy.sizes() == torch::IntArrayRef({4672}));
    REQUIRE(std::isfinite(evals[i].value.item<float>()));
  }
  REQUIRE_EQ(predictor.pendingRequestCount(), 0UL);

  // 2) queue results match a direct forward on the same weights
  for (size_t i = 0; i < evals.size(); i++) {
    Eval ref = refEval(nodes[i]);
    torch::Tensor pr =
        ref.policy.to(torch::kCPU).reshape({-1}).contiguous();
    torch::Tensor pg = evals[i].policy.reshape({-1});
    REQUIRE(pr.sizes() == pg.sizes());
    REQUIRE(std::fabs(evals[i].value.item<float>() -
                    ref.value.item<float>()) < 5e-3);
    REQUIRE(pr.allclose(pg, 1e-2, 1e-3));
  }

  // 3) publishWeights changes inference results
  {
    Predictor p2(dev);
    Node *n = nodes[0];
    Eval before = p2.enqueue(n)->future.get();
    DNN other;
    other->to(dev);
    p2.publishWeights(other);
    Eval after = p2.enqueue(n)->future.get();
    REQUIRE(!before.policy.allclose(after.policy, 1e-6, 1e-8));
    REQUIRE(std::fabs(before.value.item<float>() -
                    after.value.item<float>()) > 1e-4);
  }

  // 4) many threads racing enqueue: all futures resolve, all values finite
  {
    std::vector<std::thread> threads;
    std::vector<std::shared_ptr<Request>> reqs2;
    std::mutex m;
    for (int t = 0; t < 8; t++) {
      threads.emplace_back([&] {
        for (Node *n : nodes) {
          std::shared_ptr<Request> r = predictor.enqueue(n);
          std::lock_guard<std::mutex> lk(m);
          reqs2.push_back(r);
        }
      });
    }
    for (auto &th : threads) th.join();
    REQUIRE_EQ(reqs2.size(), 8UL * nNodes);
    for (auto &r : reqs2) {
      Eval e = r->future.get();
      REQUIRE(std::isfinite(e.value.item<float>()));
      REQUIRE(e.policy.sizes() == torch::IntArrayRef({4672}));
    }
    REQUIRE_EQ(predictor.pendingRequestCount(), 0UL);
  }

  // 5) stop() drains the queue: pending futures still resolve
  {
    Predictor p3(dev);
    std::vector<std::shared_ptr<Request>> reqs3;
    for (size_t i = 0; i < 3; i++) {
      reqs3.push_back(p3.enqueue(nodes[i]));
    }
    p3.stop();
    for (auto &r : reqs3) {
      std::future_status st = r->future.wait_for(std::chrono::seconds(5));
      REQUIRE(st == std::future_status::ready);
      REQUIRE(std::isfinite(r->future.get().value.item<float>()));
    }
    REQUIRE_EQ(p3.pendingRequestCount(), 0UL);
  }

  for (Node *n : nodes) delete testutil::chainTop(n);
  printf("test_predictor OK\n");
  return 0;
}
