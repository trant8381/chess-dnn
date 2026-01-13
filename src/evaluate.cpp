#include "evaluate.h"
#include "constants.h"
#include "create_state_fast.h"
#include "dnn.h"
#include "mcts.h"
#include "node.h"
#include <c10/core/DeviceType.h>
#include <concurrent_queue.h>

void evaluate(ConcurrentQueue<Node *> &q, DNN &model,
              std::array<GlobalData *, PARALLEL_GAMES> &g) {
  std::vector<Node *> batch;
  batch.reserve(q.size_approx());
  q.try_dequeue_bulk(batch.begin(), q.size_approx());

  #if HAS_CUDA
  torch::Tensor state = createStateFast(batch, torch::kCUDA);
  Eval outputs = model->forward(state);
  
  for (Node *node : batch) {
    GlobalData* data = g[node->threadIndex];
    putBatch(node, outputs, *data);
    data->simulation += 1;
  }
  #endif
}