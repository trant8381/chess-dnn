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
  q.try_dequeue_bulk(std::begin(batch), sizeApprox);
  std::cout << "begin batch" << std::endl;
  for (Node* node: batch) {
    if (node != nullptr) { 
      std::cout << node << std::endl;
    }
  }
  std::cout << std::end(batch) << std::endl;
  #if HAS_CUDA
  for (Node* node: batch) {
    std::cout << node->position << std::endl;
  }

  auto begin = std::begin(batch);
  auto end = std::begin(batch) + sizeApprox;

  torch::Tensor state = createStateFast(begin, end, torch::kCUDA);
  Eval outputs = model->forward(state);
  
  for (Node *node = *begin; node != *end; node++) {
    GlobalData* data = g[node->threadIndex];
    putBatch(node, outputs, *data);
    data->simulation += 1;
  }
  #endif
}