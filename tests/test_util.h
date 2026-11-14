#pragma once
// Shared test helpers for the chess-dnn test suite.
#include "create_state.h"
#include "constants.h"
#include "dnn.h"
#include "mcts.h"
#include "move_gen.h"
#include "node.h"
#include "predictor.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace testutil {

// Minimal REQUIRE macros: print the failing expression and exit nonzero.
#define REQUIRE(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);     \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

#define REQUIRE_EQ(a, b)                                                         \
  do {                                                                         \
    auto _va = (a);                                                            \
    auto _vb = (b);                                                            \
    if (!(_va == _vb)) {                                                       \
      std::fprintf(stderr, "FAIL %s:%d: %s == %s\n", __FILE__, __LINE__,       \
                   #a, #b);                                                    \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

// GPU device when available, else CPU. argv may force "cpu" / "gpu".
inline torch::Device pickDevice(int argc, char **argv) {
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "cpu") return torch::Device(torch::kCPU);
    if (a == "gpu") return torch::Device(torch::kCUDA, 0);
  }
  if (torch::cuda::is_available()) return torch::Device(torch::kCUDA, 0);
  return torch::Device(torch::kCPU);
}

// Same square<->uci convention as play.cpp / policyIndex.
inline std::string uci(const Midnight::Move &move) {
  std::string text;
  text += static_cast<char>('a' + move.from() % 8);
  text += static_cast<char>('1' + move.from() / 8);
  text += static_cast<char>('a' + move.to() % 8);
  text += static_cast<char>('1' + move.to() / 8);
  if (move.is_promotion()) {
    text += "nbrq"[move.type() & 0b0011];
  }
  return text;
}

// Fresh isolated node (no parent, no children).
inline Node *makeNode(const std::string &fen) {
  return new Node(nullptr, {}, Midnight::Position(fen));
}

// Leaf of a chain of depth nodes (each a clone of fen). The leaf has
// ancestors; chainTop() walks to the chain root for deletion.
inline Node *makeChain(const std::string &fen, int depth) {
  Node *leaf = makeNode(fen);
  Node *cur = leaf;
  for (int i = 1; i < depth; i++) {
    Node *parent = makeNode(fen);
    parent->children.insert(cur);
    cur->parent = parent;
    cur = parent;
  }
  return leaf;
}

inline Node *chainTop(Node *leaf) {
  while (leaf->parent) leaf = leaf->parent;
  return leaf;
}

}  // namespace testutil
