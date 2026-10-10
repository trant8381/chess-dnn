#include "mcts.h"
#include "constants.h"
#include "create_state.h"
#include "dnn.h"
#include "move_gen.h"
#include <ATen/core/interned_strings.h>
#include <ATen/ops/zero.h>
#include <algorithm>
#include <c10/core/DeviceType.h>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef HAS_CUDA
#include "create_state_fast.h"
#endif

// Maximum number of positions sent to the network in one forward pass.
static constexpr int BATCH_SIZE = 32;
// "N" in the paper: maximum number of tree descents used to try to fill a batch.
static constexpr int MAX_DESCENTS = 500;

struct Statistics {
  float *totalValue;
  uint32_t *visitCount;

  Statistics(float *_totalValue, uint32_t *_visitCount)
      : totalValue(_totalValue), visitCount(_visitCount) {}
};

// Convention used everywhere below:
//   * totalValue / visitCount of a node hold values seen from the point of view
//     of the SIDE TO MOVE AT THAT NODE.
//   * a node's visitCount is therefore both N(s) when the node is the parent
//     in the bandit and N(s,a) when it is the child.
//   * every node is updated exactly once per descent, by its own call.
Statistics getTreeStats(Node *node, bool isBatch, GlobalData &g) {
  if (isBatch) {
    // Lazily reset the batch-tree statistics once per batch.
    if (node->batchNum < g.currBatchNum) {
      node->batchNum = g.currBatchNum;
      node->batch_totalValue = node->totalValue;
      node->batch_visitCount = node->visitCount;
    }
    return Statistics(&node->batch_totalValue, &node->batch_visitCount);
  }
  return Statistics(&node->totalValue, &node->visitCount);
}

// creates a vector of Move with some hacks to get aorund templates.
std::vector<Midnight::Move> createMovelistVec(Midnight::Position board) {
  // Copy the moves while the MoveList is still alive. It keeps its moves
  // inside the object, so pointers taken from it dangle once the block that
  // declares it ends (the previous version read them after that point).
  if (board.turn() == Midnight::WHITE) {
    Midnight::MoveList<Midnight::WHITE> movelist(board);
    return std::vector<Midnight::Move>(movelist.begin(), movelist.end());
  }
  Midnight::MoveList<Midnight::BLACK> movelist(board);
  return std::vector<Midnight::Move>(movelist.begin(), movelist.end());
}

void playMove(Midnight::Position &board, const Midnight::Move move) {
  if (board.turn() == WHITE) {
    board.play<WHITE>(move);
  } else {
    board.play<BLACK>(move);
  }
}

// checks whether the position has insufficient material.
bool insufficientMaterial(Midnight::Position &board) {
  for (Midnight::PieceType pieceType :
       {Midnight::PAWN, Midnight::ROOK, Midnight::QUEEN}) {
    for (Midnight::Color color : {Midnight::WHITE, Midnight::BLACK}) {
      if (board.pieces[pieceType + color * 8] != 0) {
        return false;
      }
    }
  }

  int wb = __builtin_popcountll(board.pieces[Midnight::WHITE_BISHOP]);
  int bb = __builtin_popcountll(board.pieces[Midnight::BLACK_BISHOP]);
  int wn = __builtin_popcountll(board.pieces[Midnight::WHITE_KNIGHT]);
  int bn = __builtin_popcountll(board.pieces[Midnight::BLACK_KNIGHT]);
  if (wn + wb + bn + bb == 0)
    return true;

  // King + minor vs king
  if ((wn + wb == 1) && (bn + bb == 0))
    return true;

  if ((bn + bb == 1) && (wn + wb == 0))
    return true;

  // King + bishop vs king + bishop (same color bishops)
  if (wb == 1 && bb == 1 && wn == 0 && bn == 0) {
    // Get bishop square colors
    int w_sq = __builtin_ctzll(board.pieces[Midnight::WHITE_BISHOP]);
    int b_sq = __builtin_ctzll(board.pieces[Midnight::BLACK_BISHOP]);

    bool w_dark = ((w_sq ^ (w_sq >> 3)) & 1);
    bool b_dark = ((b_sq ^ (b_sq >> 3)) & 1);

    return w_dark == b_dark;
  }

  return false;
}

// check if the board state is terminal
bool isTerminal(Midnight::Position &board) {
  std::vector<Midnight::Move> movelist = createMovelistVec(board);

  if (movelist.size() == 0 ||
      board.has_repetition(Midnight::Position::THREE_FOLD) ||
      board.fifty_move_rule() >= 100)
    return true;

  if (insufficientMaterial(board))
    return true;

  return false;
}

// returns the board state's terminal value if board is terminal.
float terminalValue(Midnight::Position &board) {
  uint64_t whiteKingBoard = board.pieces[Midnight::WHITE_KING];
  uint64_t blackKingBoard = board.pieces[Midnight::BLACK_KING];

  if (createMovelistVec(board).size() == 0) {
    if (board.turn() == Midnight::WHITE &&
        board.attackers_of<Midnight::BLACK>(
            Midnight::Square(__builtin_ctzll(whiteKingBoard)),
            whiteKingBoard)) {
      return -1.0f;
    } else if (board.turn() == Midnight::BLACK &&
               board.attackers_of<Midnight::WHITE>(
                   Midnight::Square(__builtin_ctzll(blackKingBoard)),
                   blackKingBoard)) {
      return -1.0f;
    } else {
      return 0.0f;
    }
  }

  if (board.has_repetition(Midnight::Position::THREE_FOLD) ||
      board.fifty_move_rule() >= 100 || insufficientMaterial(board)) {
    return 0.0f;
  }

  assert(false);
  return 0.0f;
}

static const int KDX[8] = {1, 2, 2, 1, -1, -2, -2, -1};
static const int KDY[8] = {2, 1, -1, -2, -2, -1, 1, 2};

// returns the index for the policy tensor that corresponds to a move on the
// board.
//
// The return type was changed from float to int (it is used as a tensor
// index). The body is deliberately unchanged because it has to match the
// encoding the network was trained with. Things to verify against your
// training pipeline:
//   1. direction is in 1..8, so direction * 7 + distance spans 7..62 and the
//      direction == 8 (down-left) moves collide with the knight planes
//      56..63. (direction - 1) * 7 + distance would give 0..55.
//   2. for black, direction - 3 is 3..5, so 64 + piece * 3 + 3..5 can run past
//      plane 72 into the next square's planes.
//   3. (move.type() & 0b0111) != PR_QUEEN is always true if PR_QUEEN has the
//      high bit set (0b1011 in Surge-style move flags), which would encode
//      queen promotions as underpromotions. (type & 0b0011) != 0b0011 is the
//      usual test.
int policyIndex(Position &board, Move move) {
  Piece piece = board.piece_at(move.from());
  PieceType pieceType = static_cast<PieceType>(piece % 8);
  int diffx = move.to() % 8 - move.from() % 8;
  int diffy = move.to() / 8 - move.from() / 8;
  int moveType = 0;

  if (pieceType == KNIGHT) {
    for (int i = 0;; i++) {
      if (diffx == KDX[i] && diffy == KDY[i]) {
        moveType = 56 + i;
        break;
      }
    }
  } else {
    int distance = std::max(abs(diffx), abs(diffy)) - 1;
    int direction = (diffx == 0  ? 0
                     : diffx > 0 ? 1
                                 : 2) +
                    3 * (diffy == 0  ? 0
                         : diffy > 0 ? 1
                                     : 2);
    if (move.is_promotion() && ((move.type() & 0b0111) != PR_QUEEN)) {
      moveType = 64 + (move.type() & 0b0011) * 3 + direction - 3;
    } else {
      moveType = direction * 7 + distance;
    }
  }

  return move.from() * 73 + moveType;
}

// Adds one real visit to `node`. `value` is from the point of view of the side
// to move at `node`.
static void backup(Node *node, float value, bool batchMode, GlobalData &g) {
  Statistics s = getTreeStats(node, batchMode, g);
  *s.totalValue += value;
  *s.visitCount += 1;
}

// Virtual mean (batch tree only): add VL visits that carry the node's current
// mean, so its mean stays the same but its visit count grows. This pushes the
// next descent of the same batch towards other leaves. `fallbackMean` is used
// for a node that has no statistics yet.
static void virtualMean(Node *node, float fallbackMean, GlobalData &g) {
  Statistics s = getTreeStats(node, true, g);
  float mean = *s.visitCount > 0 ? *s.totalValue / *s.visitCount : fallbackMean;
  *s.totalValue += VL * mean;
  *s.visitCount += VL;
}

// One descent. Returns the value of `node` from the point of view of the side
// to move at `node`, or UNKNOWN if the descent ended on a position that has
// not been evaluated by the network yet.
//
// batchMode == true : descent in the scratch (batch) statistics. Unknown leaves
//                     are queued for the next forward pass and penalised with
//                     the virtual mean.
// batchMode == false: descent in the real statistics. Unknown leaves change
//                     nothing.
//
// `ownMean` is the parent's estimate of this node's mean (from this node's own
// point of view). It is only used by the virtual mean of an unvisited leaf.
float batchPUCT(Node *node, bool batchMode, GlobalData &g,
                float ownMean = 0.0f) {
  Batch &batch = g.batch;

  if (isTerminal(node->position)) {
    float value = terminalValue(node->position);
    backup(node, value, batchMode, g);
    return value;
  }

  Statistics stats = getTreeStats(node, batchMode, g);

  // A node is part of a tree once that tree has recorded a visit for it and it
  // has been evaluated.
  bool inTree = *stats.visitCount > 0 && node->valueEval != INFINITY;

  if (!inTree) {
    if (node->valueEval == INFINITY) {
      // Not evaluated by the network yet.
      if (batchMode) {
        if (std::find(batch.nodes.begin(), batch.nodes.end(), node) ==
            batch.nodes.end()) {
          batch.nodes.push_back(node);
          if (g.device == torch::kCPU) {
            batch.nnInputs.push_back(
                createState(constructHistory(node), g.device));
          }
        }
        virtualMean(node, ownMean, g);
      }
      return UNKNOWN;
    }

    // Already evaluated (its children exist) but this tree has not counted it
    // yet: attach it and return its value.
    if (!batchMode) {
      node->initialized = true;
    }
    backup(node, node->valueEval, batchMode, g);
    return node->valueEval;
  }

  const float nodeVisits = static_cast<float>(*stats.visitCount);
  // muFPU: unvisited moves are worth the average value of the node.
  const float fpu = *stats.totalValue / nodeVisits;

  float bestScore = -INFINITY;
  float bestQ = fpu;
  Node *selected = nullptr;

  for (Node *childNode : node->children) {
    Statistics childStats = getTreeStats(childNode, batchMode, g);

    // The child stores values from the opponent's point of view.
    float q = fpu;
    if (*childStats.visitCount > 0) {
      q = -(*childStats.totalValue / *childStats.visitCount);
    }

    float bandit = q + C_PUCT * childNode->policyEval * sqrtf(nodeVisits) /
                           (1.0f + *childStats.visitCount);

    if (bandit > bestScore) {
      bestScore = bandit;
      bestQ = q;
      selected = childNode;
    }
  }

  if (selected == nullptr) {
    // Either the node has no children, or every bandit score was NaN.
    std::ostringstream msg;
    msg << "batchPUCT: no move can be selected at " << node->position.fen()
        << " (children=" << node->children.size() << ", visits=" << nodeVisits
        << ", mean=" << fpu << ", valueEval=" << node->valueEval
        << ", batchMode=" << batchMode << ")";
    throw std::runtime_error(msg.str());
  }

  float res = batchPUCT(selected, batchMode, g, -bestQ);

  if (res == UNKNOWN) {
    if (batchMode) {
      virtualMean(node, ownMean, g);
    }
    return UNKNOWN;
  }

  float value = -res;
  backup(node, value, batchMode, g);
  return value;
}

void getBatch(Node *root, GlobalData &g, int maxBatch) {
  Batch &batch = g.batch;

  for (int i = 0;
       i < MAX_DESCENTS && static_cast<int>(batch.nodes.size()) < maxBatch;
       i++) {
    batchPUCT(root, true, g);
  }
}

void putBatch(Node *root, Eval &outputs, GlobalData &g) {
  Batch &batch = g.batch;
  std::vector<Node *> &nodes = batch.nodes;

  torch::Tensor policyCpu =
      outputs.policy.to(torch::kCPU, torch::kFloat32)
          .reshape({static_cast<int64_t>(nodes.size()), -1})
          .contiguous();
  torch::Tensor valueCpu = outputs.value.to(torch::kCPU, torch::kFloat32)
                               .reshape({-1})
                               .contiguous();
  auto policy = policyCpu.accessor<float, 2>();
  auto value = valueCpu.accessor<float, 1>();

  for (size_t i = 0; i < nodes.size(); i++) {
    Node *node = nodes[i];
    std::vector<Midnight::Move> moves = createMovelistVec(node->position);

    if (moves.empty() || !std::isfinite(value[i])) {
      throw std::runtime_error(
          "putBatch: unusable network output at " + node->position.fen() +
          " (value " + std::to_string(value[i]) + ", " +
          std::to_string(moves.size()) +
          " legal moves). If the weights diverged, check the training loss.");
    }

    std::vector<float> priors(moves.size());
    float maxLogit = -INFINITY;
    for (size_t k = 0; k < moves.size(); k++) {
      priors[k] = policy[i][policyIndex(node->position, moves[k])];
      maxLogit = std::max(maxLogit, priors[k]);
    }
    float sum = 0.0f;
    for (float &p : priors) {
      if (POLICY_IS_LOGITS) {
        p = std::exp(p - maxLogit);
      }
      sum += p;
    }
    if (!std::isfinite(sum)) {
      throw std::runtime_error("putBatch: non-finite policy output at " +
                               node->position.fen());
    }

    for (size_t k = 0; k < moves.size(); k++) {
      float prior = sum > 0.0f ? priors[k] / sum : 1.0f / moves.size();

      Midnight::Position newBoard(node->position);
      playMove(newBoard, moves[k]);

      node->children.insert(new Node(node, {}, newBoard, prior));
    }

    node->valueEval = value[i];
  }

  batch.nnInputs.clear();
  batch.nodes.clear();

  for (int i = 0; i < MAX_DESCENTS; i++) {
    if (batchPUCT(root, false, g) == UNKNOWN) {
      break;
    }
  }
}

// Picks the move to play from the root visit counts.
static Node *selectMove(Node *root, float temperature) {
  if (root->children.empty()) {
    return nullptr;
  }

  std::vector<Node *> kids(root->children.begin(), root->children.end());

  uint32_t maxVisits = 0;
  for (Node *child : kids) {
    maxVisits = std::max(maxVisits, child->visitCount);
  }

  if (maxVisits == 0) {
    return *std::max_element(kids.begin(), kids.end(), [](Node *a, Node *b) {
      return a->policyEval < b->policyEval;
    });
  }

  if (temperature <= 1e-3f) {
    return *std::max_element(kids.begin(), kids.end(), [](Node *a, Node *b) {
      return a->visitCount < b->visitCount;
    });
  }

  std::vector<double> weights(kids.size());
  for (size_t k = 0; k < kids.size(); k++) {
    weights[k] = std::pow(static_cast<double>(kids[k]->visitCount) / maxVisits,
                          1.0 / temperature);
  }

  thread_local std::mt19937_64 rng{std::random_device{}()};
  std::discrete_distribution<size_t> dist(weights.begin(), weights.end());
  return kids[dist(rng)];
}

Sample makeSample(Node *root) {
  Sample sample;
  sample.value = 0.0f;
  sample.input = createState(constructHistory(root), torch::kCPU)
                     .reshape({INPUT_PLANES, 8, 8})
                     .contiguous();

  uint32_t total = 0;
  for (Node *child : root->children) {
    total += child->visitCount;
  }
  if (total == 0) {
    return sample;
  }

  for (const Midnight::Move &move : createMovelistVec(root->position)) {
    Midnight::Position next(root->position);
    playMove(next, move);

    for (Node *child : root->children) {
      if (child->position.hash() == next.hash()) {
        sample.policy.push_back({policyIndex(root->position, move),
                                 static_cast<float>(child->visitCount) / total});
        break;
      }
    }
  }

  return sample;
}

Node *getNextMove(Node *root, DNN &model, float temperature, GlobalData &g) {
  Batch &batch = g.batch;

  if (isTerminal(root->position)) {
    return nullptr;
  }

  int evaluations = 0;
  while (evaluations < SIMULATIONS) {
    g.currBatchNum += 1;

    getBatch(root, g, std::min<int>(BATCH_SIZE, SIMULATIONS - evaluations));
    if (batch.nodes.empty()) {
      break;
    }

    torch::Tensor batchedInput;
    if (g.device == torch::kCPU) {
      batchedInput = torch::zeros({static_cast<long>(batch.nnInputs.size()),
                                   INPUT_PLANES, 8, 8})
                         .to(g.device);
      for (size_t j = 0; j < batch.nnInputs.size(); j++) {
        batchedInput[j] = batch.nnInputs[j].to(g.device);
      }
    } else {
#ifdef HAS_CUDA
      batchedInput = createStateFast(batch.nodes, g.device);
#endif
    }

    evaluations += static_cast<int>(batch.nodes.size());

    Eval outputs = model->forward(batchedInput);
    putBatch(root, outputs, g);
  }

  return selectMove(root, temperature);
}