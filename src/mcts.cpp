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
#include <vector>
#ifdef HAS_CUDA
#include "create_state_fast.h"
#endif

struct Statistics {
  float *totalValue;
  uint32_t *visitCount;

  Statistics(float *_totalValue, uint32_t *_visitCount)
      : totalValue(_totalValue), visitCount(_visitCount) {}
};

Statistics getTreeStats(Node *node, bool isBatch, GlobalData &g) {
  if (node->batchNum < g.currBatchNum) {
    node->batchNum = g.currBatchNum;
    node->batch_totalValue = node->totalValue;
    node->batch_visitCount = node->visitCount;
  }
  if (isBatch) {
    return Statistics(&node->batch_totalValue, &node->batch_visitCount);
  } else {
    return Statistics(&node->totalValue, &node->visitCount);
  }
}

// creates a vector of Move with some hacks to get aorund templates.
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
float policyIndex(Position &board, Move move) {
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

void updateStatistics(float res, Node *node, Node *childNode, bool batch,
                      GlobalData &g) {
  Statistics parentStats = getTreeStats(node, batch, g);
  Statistics childStats = getTreeStats(childNode, batch, g);

  if (res != UNKNOWN) {
    *parentStats.totalValue += res;
    *parentStats.visitCount += 1;
    *childStats.totalValue += res;
    *childStats.visitCount += 1;
  }
}

void updateStatisticsGet(float res, Node *node, Node *childNode, bool batch,
                         GlobalData &g) {
  if (res == UNKNOWN) {
    Statistics parentStats = getTreeStats(node, batch, g);
    Statistics childStats = getTreeStats(childNode, batch, g);
    float mean;
    if (*childStats.visitCount == 0) {
      mean = FPU;
    } else {
      mean = *childStats.totalValue / *childStats.visitCount;
    }

    *childStats.totalValue += VL * mean;
    *childStats.visitCount += VL;
    *parentStats.totalValue += VL * mean;
    *parentStats.visitCount += VL;
  } else {
    updateStatistics(res, node, childNode, batch, g);
  }
}

float batchPUCT(Node *node, bool getBatch, GlobalData &g) {
  Batch &batch = g.batch;
  if (isTerminal(node->position)) {
    return terminalValue(node->position);
  }

  if (!node->initialized) {
    if (getBatch) {
      batch.nodes.push_back(node);
      if (g.device == torch::kCPU) {
        batch.nnInputs.push_back(createState(constructHistory(node), g.device));
      }
    } else if (node->valueEval != INFINITY) {
      node->initialized = true;
      return node->valueEval;
    }
    return UNKNOWN;
  }

  float bestScore = -INFINITY;
  Node *selected = nullptr;

  for (Node *childNode : node->children) {
    Statistics childStats = getTreeStats(childNode, getBatch, g);
    Statistics nodeStats = getTreeStats(node, getBatch, g);
    float mean = FPU;

    if (*childStats.visitCount > 0) {
      mean = *childStats.totalValue / *childStats.visitCount;
    }
    // std::cout << mean << " " << childNode->policyEval << " " <<
    // *nodeStats.visitCount << " " << *childStats.visitCount << " " <<
    // *childStats.totalValue << std::endl;
    float bandit = mean + C_PUCT * childNode->policyEval *
                              (sqrtf(*nodeStats.visitCount) /
                               (1 + *childStats.visitCount));

    if (bandit > bestScore) {
      bestScore = bandit;
      selected = childNode;
    }
  }

  float res = batchPUCT(selected, getBatch, g);

  updateStatisticsGet(res, node, selected, getBatch, g);
  if (res != UNKNOWN) {
    return -res;
  }
  return UNKNOWN;
}

void getBatch(Node *node, GlobalData &g) {
  Batch &batch = g.batch;

  for (int i = 0; i < 32; i++) {
    batchPUCT(node, true, g);
    if (batch.nodes.size() >= 2 &&
        batch.nodes[batch.nodes.size() - 1]->position.hash() ==
            batch.nodes[batch.nodes.size() - 2]->position.hash()) {
      batch.nodes.pop_back();
      if (g.device == torch::kCPU) {
        batch.nnInputs.pop_back();
      }
      break;
    }
  }
}

void putBatch(Node *node, Eval &outputs, GlobalData &g) {
  Batch &batch = g.batch;
  std::vector<Node *> &nodes = batch.nodes;

  for (size_t i = 0; i < nodes.size(); i++) {
    for (Move move : createMovelistVec(nodes[i]->position)) {
      Midnight::Position newBoard(nodes[i]->position);
      playMove(newBoard, move);

      Node *childNode =
          new Node(nodes[i], {}, newBoard,
                   outputs.policy[i][policyIndex(nodes[i]->position, move)]
                       .item()
                       .toFloat());
      childNode->threadIndex = node->threadIndex;

      nodes[i]->children.insert(childNode);
    }

    nodes[i]->valueEval = outputs.value[i].item().toFloat();
  }

  while (true) {
    float result = batchPUCT(node, false, g);

    if (result == UNKNOWN) {
      break;
    }
  }
}

Node *getNextMove(Node *node, DNN &model, float temperature, GlobalData &g) {
  Batch &batch = g.batch;

  while (g.simulation < SIMULATIONS) {
    getBatch(node, g);
    if (batch.nodes.size() == 0) {
      continue;
    }

    torch::Tensor batchedInput;
    if (g.device == torch::kCPU) {
      batchedInput = torch::zeros({static_cast<long>(batch.nnInputs.size()),
                                   INPUT_PLANES, 8, 8})
                         .to(g.device);
      for (size_t j = 0; j < batch.nnInputs.size(); j++) {
        batchedInput[j] = batch.nnInputs[j].to(g.device);
      }
      g.simulation += batch.nodes.size();

      Eval outputs = model->forward(batchedInput);
      putBatch(node, outputs, g);
    } else {
      #ifdef HAS_CUDA
      g.q->enqueue_bulk(g.batch.nodes.begin(), g.batch.nodes.size());
      #endif
    }
    batch.nnInputs = {};
    batch.nodes = {};
  }

  g.currBatchNum += 1;
  g.simulation = 0;

  float total = 0;
  for (Node *child : node->children) {
    total += pow(child->visitCount, 1.0 / temperature);
    std::cout << child->visitCount << std::endl;
  }
  int i = rand() % static_cast<int>(total);
  float curr = 0;

  for (Node *child : node->children) {
    curr += pow(child->visitCount, 1.0 / temperature);
    if (curr >= i) {
      return child;
    }
  }

  assert(false);
}