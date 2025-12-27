#include "dnn.h"
#include "move_gen.h"
#include <cstdint>
#include <vector>

struct Node {
  Node *parent;
  std::vector<Node> children;
  Midnight::Position position;
  float totalValue;
  uint32_t visitCount;
  float meanValue;
  float policyEval;

  Node(Node *_parent, const std::vector<Node> _children,
       const Midnight::Position _position, float _policyEval) {
    parent = _parent;
    children = _children;
    position = _position;
    totalValue = 0;
    visitCount = 0;
    meanValue = 0;
    policyEval = _policyEval;
  }

  Node(const std::vector<Node> _children, const Midnight::Position _position,
       float _policyEval) {
    parent = nullptr;
    children = _children;
    position = _position;
    totalValue = 0;
    visitCount = 0;
    meanValue = 0;
    policyEval = _policyEval;
  }
};

bool isTerminal(Midnight::Position &board);
float simulate(Node *node, DNN &model, const torch::Device &device);
Node playMove(Node *root, DNN &model, const torch::Device &device,
              float temperature);