#include "evaluate.h"
#include "constants.h"
#include "create_state_fast.h"
#include "dnn.h"
#include "mcts.h"
#include "node.h"
#include <algorithm>
#include <c10/core/DeviceType.h>
#include <concurrent_queue.h>
#include <iterator>

void evaluate(ConcurrentQueue<Node *> &q, DNN &model,
              std::array<GlobalData *, PARALLEL_GAMES> &g) {
  Node* batch[512];
  int sizeApprox = std::min(q.size_approx(), 512UL);
  if (sizeApprox == 0) {
    return;
  }
  q.try_dequeue_bulk(std::begin(batch), sizeApprox);

  std::cout << std::end(batch) << std::endl;
  // #if HAS_CUDA
  auto begin = std::begin(batch);
  auto end = std::begin(batch) + sizeApprox;

  torch::Tensor state = createStateFast(begin, end, torch::kCUDA);
  Eval outputs = model->forward(state);
  
  for (Node *node = *begin; node != *end; node++) {
    GlobalData* data = g[node->threadIndex];
    data->batch.nodes.push_back(node);
    data->simulation += 1;
  }

  for (GlobalData* data : g) {
    putBatch(nullptr, outputs, *data);
    data->simulation += data->batch.nodes.size();
    data->batch.nodes = {};
  }
  // #endif
}