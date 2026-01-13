#include "dnn.h"
#include "mcts.h"
#include "node.h"
#include "concurrent_queue.h"

using namespace moodycamel;

void evaluate(ConcurrentQueue<Node*> &data, DNN &model, std::array<GlobalData *, PARALLEL_GAMES> &g);