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

  Node(Node *_parent, const std::vector<Node> _children,
       const Midnight::Position _position) {
    parent = _parent;
    children = _children;
    visitTotal = 0;
    visitCount = 0;
    visitMean = 0;
  }

  Node(const std::vector<Node> _children, const Midnight::Position _position) {
    parent = nullptr;
    children = _children;
    position = _position;
    visitTotal = 0;
    visitCount = 0;
    visitMean = 0;
  }
};