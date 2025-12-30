#include "dnn.h"
#include "move_gen.h"
#include <cstdint>
#include <mutex>
#include <vector>

static constexpr int UNVISITED = 0;
static constexpr int EVALUTING = 1;
static constexpr int DONE = 2;

struct Node {
  Node *parent;
  std::vector<Node*> children;
  Midnight::Position position;
  float totalValue = 0;
  uint32_t visitCount = 0;
  uint32_t virtualLoss = 0;
  float meanValue = 0;
  float policyEval;
  int state = UNVISITED;
  std::mutex mutex;

  Node(Node *_parent, const std::vector<Node*> _children,
       const Midnight::Position _position, float _policyEval) {
    parent = _parent;
    children = _children;
    position = _position;
    policyEval = _policyEval;
  }

  Node(const std::vector<Node*> _children, const Midnight::Position _position,
       float _policyEval) {
    parent = nullptr;
    children = _children;
    position = _position;
    policyEval = _policyEval;
  }

  ~Node() {
    for (Node* node : children) {
      delete node;
    }
  }
};

std::vector<Midnight::Move> createMovelistVec(Midnight::Position board);
bool isTerminal(Midnight::Position &board);
float simulate(Node *node);
Node *playMove(Node *root, DNN &model, const torch::Device &device,
              float temperature);