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
#include <unordered_map>
#include <vector>

Node *transpositionTable[TABLE_SIZE] = {};
std::unordered_map<uint64_t, Node *> tree;
struct Batch {
  std::vector<torch::Tensor> nnInputs;
  std::vector<Node *> nodes;
};
uint32_t currBatchNum = 0;
Batch batch;

struct Statistics {
  float *totalValue;
  uint32_t *visitCount;

  Statistics(float *_totalValue, uint32_t *_visitCount)
      : totalValue(_totalValue), visitCount(_visitCount) {}
};

Statistics getTreeStats(Node *node, bool isBatch) {
  if (node->batchNum < currBatchNum) {
    node->batchNum = currBatchNum;
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

// creates an array of the history boards.
std::array<Position, HISTORY_BOARDS> constructHistory(Node *node) {
  std::array<Position, HISTORY_BOARDS> history = {
      Position(START_FEN), Position(START_FEN), Position(START_FEN),
      Position(START_FEN)};

  const Node *current = node;
  for (int i = 0; i < 4 && current != nullptr; i++) {
    history[i] = current->position;
    current = current->parent;
  }

  return history;
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

void updateStatistics(float res, Node *node, Node *childNode, bool batch) {
  if (res != UNKNOWN && batch) {
    node->batch_totalValue += res;
    node->batch_visitCount += 1;
    childNode->batch_totalValue += res;
    childNode->batch_visitCount += 1;
  } else if (res != UNKNOWN) {
    node->totalValue += res;
    node->visitCount += 1;
    childNode->totalValue += res;
    childNode->visitCount += 1;
  }
}

void updateStatisticsGet(float res, Node *node, Node *childNode, bool batch) {
  if (res == UNKNOWN) {
    Statistics parentStats = getTreeStats(node, batch);
    Statistics childStats = getTreeStats(childNode, batch);
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
    updateStatistics(res, node, childNode, batch);
  }
}

float batchPUCT(Node *node, bool getBatch) {
  if (isTerminal(node->position)) {
    return terminalValue(node->position);
  }

  if (node->children.size() == 0) {
    if (getBatch) {
      batch.nodes.push_back(node);
      batch.nnInputs.push_back(
          createState(constructHistory(node), torch::kCPU));
    }
    return UNKNOWN;
  }

  float bestScore = -INFINITY;
  Node *selected = nullptr;

  for (Node *childNode : node->children) {
    Statistics childStats = getTreeStats(childNode, getBatch);
    Statistics nodeStats = getTreeStats(node, getBatch);
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

  float res = batchPUCT(selected, getBatch);

  if (getBatch) {
    updateStatisticsGet(res, node, selected, getBatch);
  } else {
    updateStatistics(res, node, selected, getBatch);
  }

  return res;
}

void getBatch(Node *node) {
  for (int i = 0; i < 32; i++) {
    batchPUCT(node, true);
    if (batch.nodes.size() >= 2 &&
        batch.nodes[batch.nodes.size() - 1]->position.hash() ==
            batch.nodes[batch.nodes.size() - 2]->position.hash()) {
      batch.nodes.pop_back();
      batch.nnInputs.pop_back();
      break;
    }
  }
}

void putBatch(Node *node, std::vector<Node *> &nodes, Eval &outputs) {
  std::cout << batch.nodes.size() << std::endl;
  for (size_t i = 0; i < nodes.size(); i++) {
    for (Move move : createMovelistVec(nodes[i]->position)) {
      Midnight::Position newBoard(nodes[i]->position);
      playMove(newBoard, move);

      Node *childNode =
          new Node(nodes[i], {}, newBoard,
                   outputs.policy[i][policyIndex(nodes[i]->position, move)]
                       .item()
                       .toFloat());

      nodes[i]->children.insert(childNode);
    }

    nodes[i]->valueEval = outputs.value[i].item().toFloat();
    transpositionTable[nodes[i]->position.hash() % TABLE_SIZE] = nodes[i];
  }

  batch.nnInputs = {};
  batch.nodes = {};

  while (true) {
    float result = batchPUCT(node, false);

    if (result == UNKNOWN) {
      break;
    }
  }
}

Node *getNextMove(Node *node, DNN &model) {
  for (int i = 0; i < SIMULATIONS;) {
    getBatch(node);
    torch::Tensor batchedInput = torch::zeros(
        {static_cast<long>(batch.nnInputs.size()), INPUT_PLANES, 8, 8});
    for (size_t j = 0; j < batch.nnInputs.size(); j++) {
      batchedInput[j] = batch.nnInputs[j];
    }
    i += batch.nnInputs.size();

    Eval outputs = model->forward(batchedInput);
    putBatch(node, batch.nodes, outputs);
  }

  Node *selected = *node->children.begin();

  for (Node *child : node->children) {
    if (child->visitCount > selected->visitCount) {
      selected = child;
    }
  }

  return selected;
}