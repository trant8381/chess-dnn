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
#include <vector>
#ifdef HAS_CUDA
#include "create_state_fast.h"
#endif

static constexpr int BATCH_SIZE = 32;
static constexpr int MAX_DESCENTS = 500;

struct Statistics {
  float *totalValue;
  uint32_t *visitCount;

  Statistics(float *_totalValue, uint32_t *_visitCount)
      : totalValue(_totalValue), visitCount(_visitCount) {}
};

Statistics getTreeStats(Node *node, bool isBatch, GlobalData &g) {
  if (isBatch) {
    if (node->batchNum < g.currBatchNum) {
      node->batchNum = g.currBatchNum;
      node->batch_totalValue = node->totalValue;
      node->batch_visitCount = node->visitCount;
    }
    return Statistics(&node->batch_totalValue, &node->batch_visitCount);
  }
  return Statistics(&node->totalValue, &node->visitCount);
}

std::vector<Midnight::Move> createMovelistVec(Midnight::Position board) {
  const Midnight::Move *begin;
  const Midnight::Move *end;
  if (board.turn() == Midnight::WHITE) {
    Midnight::MoveList<Midnight::WHITE> movelist(board);
    begin = movelist.begin();
    end = movelist.end();
  } else {
    Midnight::MoveList<Midnight::BLACK> movelist(board);
    begin = movelist.begin();
    end = movelist.end();
  }

  return std::vector<Midnight::Move>(begin, end);
}

void playMove(Midnight::Position &board, const Midnight::Move move) {
  if (board.turn() == WHITE) {
    board.play<WHITE>(move);
  } else {
    board.play<BLACK>(move);
  }
}

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

  if (wn == 1 && bn == 1)
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

static void backup(Node *node, float value, bool batchMode, GlobalData &g) {
  Statistics s = getTreeStats(node, batchMode, g);
  *s.totalValue += value;
  *s.visitCount += 1;
}

static void virtualMean(Node *node, float fallbackMean, GlobalData &g) {
  Statistics s = getTreeStats(node, true, g);
  float mean = *s.visitCount > 0 ? *s.totalValue / *s.visitCount : fallbackMean;
  *s.totalValue += VL * mean;
  *s.visitCount += VL;
}


float batchPUCT(Node *node, bool batchMode, GlobalData &g,
                float ownMean = 0.0f) {
  Batch &batch = g.batch;

  if (isTerminal(node->position)) {
    float value = terminalValue(node->position);
    backup(node, value, batchMode, g);
    return value;
  }

  Statistics stats = getTreeStats(node, batchMode, g);

  bool inTree = *stats.visitCount > 0 && node->valueEval != INFINITY;

  if (!inTree) {
    if (node->valueEval == INFINITY) {
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


    if (!batchMode) {
      node->initialized = true;
    }
    backup(node, node->valueEval, batchMode, g);
    return node->valueEval;
  }

  const float nodeVisits = static_cast<float>(*stats.visitCount);
  const float fpu = *stats.totalValue / nodeVisits;

  float bestScore = -INFINITY;
  float bestQ = fpu;
  Node *selected = nullptr;

  for (Node *childNode : node->children) {
    Statistics childStats = getTreeStats(childNode, batchMode, g);

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

  assert(selected != nullptr);

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

    std::vector<float> priors(moves.size());
    float sum = 0.0f;
    for (size_t k = 0; k < moves.size(); k++) {
      priors[k] = policy[i][policyIndex(node->position, moves[k])];
      sum += priors[k];
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

static Node *selectMove(Node *root, float temperature) {
  if (root->children.empty()) {
    return nullptr;
  }

  std::vector<Node *> kids(root->children.begin(), root->children.end());

  uint32_t maxVisits = 0;
  for (Node *child : kids) {
    maxVisits = std::max(maxVisits, child->visitCount);
  }

  // Nothing was visited: fall back on the policy.
  if (maxVisits == 0) {
    return *std::max_element(kids.begin(), kids.end(), [](Node *a, Node *b) {
      return a->policyEval < b->policyEval;
    });
  }

  // (Near) zero temperature: play the most visited move.
  if (temperature <= 1e-3f) {
    return *std::max_element(kids.begin(), kids.end(), [](Node *a, Node *b) {
      return a->visitCount < b->visitCount;
    });
  }

  // Sample proportionally to visits^(1/T). Visits are normalised by the
  // maximum first so a small temperature cannot overflow.
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
    return nullptr; // game over, nothing to search
  }

  int evaluations = 0;
  while (evaluations < SIMULATIONS) {
    g.currBatchNum += 1;

    getBatch(root, g, std::min<int>(BATCH_SIZE, SIMULATIONS - evaluations));
    if (batch.nodes.empty()) {
      break; // every descent ended on a known value, nothing left to evaluate
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