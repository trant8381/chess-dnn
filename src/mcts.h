#include "move_gen.h"
#include <cstdint>
#include <vector>

struct Node {
  Node *parent;
  std::vector<Node> children;
  Midnight::Position position;
  uint32_t visitTotal;
  uint32_t visitCount;
  uint32_t visitMean;
  float policyEval;

  Node(Node *_parent, const std::vector<Node> _children,
       const Midnight::Position _position, float _policyEval) {
    parent = _parent;
    children = _children;
    position = _position;
    visitTotal = 0;
    visitCount = 0;
    visitMean = 0;
    policyEval = _policyEval;
  }

  Node(const std::vector<Node> _children, const Midnight::Position _position, float _policyEval) {
    parent = nullptr;
    children = _children;
    position = _position;
    visitTotal = 0;
    visitCount = 0;
    visitMean = 0;
    policyEval = _policyEval;
  }
};