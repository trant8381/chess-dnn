// CUDA-only: createStateFast (bit-plane kernels) must produce bit-identical
// inputs to the CPU createState encoder for the same nodes.
#include "test_util.h"
#include "create_state_fast.h"

int main() {
  REQUIRE(torch::cuda::is_available());
  const torch::Device dev(torch::kCUDA, 0);

  const std::string fens[] = {
      START_FEN,
      "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
      "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
      "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
      "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 6 4",
  };
  std::vector<Node *> nodes;
  for (const std::string &f : fens) nodes.push_back(testutil::makeNode(f));

  auto cpuPlanes = [](Node *n) {
    return createState(constructHistory(n), torch::kCPU);
  };

  // single-node calls: exact equality with the CPU encoder
  for (size_t i = 0; i < nodes.size(); i++) {
    Node *n = nodes[i];
    torch::Tensor gpu = createStateFast({n}, dev).to(torch::kCPU);
    torch::Tensor cpu = cpuPlanes(n).unsqueeze(0);
    REQUIRE(gpu.sizes() == cpu.sizes());
    const torch::Tensor diff = (gpu - cpu).abs();
    const float maxDiff = diff.max().item<float>();
    if (maxDiff != 0.0f) {
      const int64_t flat = diff.reshape({-1}).argmax().item<int64_t>();
      printf("node %zu fen=%s maxDiff=%g flat=%ld plane=%ld r=%ld f=%ld\n",
             i, fens[i].c_str(), maxDiff, static_cast<long>(flat),
             static_cast<long>(flat / 64), static_cast<long>((flat % 64) / 8),
             static_cast<long>(flat % 8));
      printf("gpu=%g cpu=%g\n",
             static_cast<double>(gpu.reshape({-1})[flat].item<float>()),
             static_cast<double>(cpu.reshape({-1})[flat].item<float>()));
    }
    REQUIRE(maxDiff == 0.0f);
  }

  // one batched call with 70 nodes (spans several 64-element kernel tiles)
  std::vector<Node *> big;
  for (int i = 0; i < 70; i++) big.push_back(nodes[i % nodes.size()]);
  torch::Tensor gpuB = createStateFast(big, dev).to(torch::kCPU);
  std::vector<torch::Tensor> singles;
  for (Node *n : big) singles.push_back(cpuPlanes(n));
  torch::Tensor cpuB = torch::stack(singles);
  REQUIRE(gpuB.sizes() == cpuB.sizes());
  REQUIRE((gpuB - cpuB).abs().max().item<float>() == 0.0f);

  for (Node *n : nodes) delete n;
  printf("test_state_fast OK\n");
  return 0;
}
