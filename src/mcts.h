#include "dnn.h"
#include "move_gen.h"

Node *getNextMove(Node *node, DNN &model);
bool isTerminal(Midnight::Position &board);