#include "mcts.h"
#include "constants.h"
#include "create_state.h"
#include "dnn.h"
#include "move_gen.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

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
  std::array<Position, HISTORY_BOARDS> history;
  Node *currNode = node;
  for (int i = 0; i < HISTORY_BOARDS; i++) {
    if (currNode == nullptr) {
      history[i] = Position(START_FEN);
    } else {
      history[i] = currNode->position;
      currNode = currNode->parent;
    }
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
  int moveType;

  if (pieceType == KNIGHT) {
    for (int i = 0; i < 8; i++) {
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

// expansion stage of MCTS.
void expand(Node *node, const torch::Tensor &policy) {
  std::vector<Move> movelist = createMovelistVec(node->position);

  for (Move move : movelist) {
    if (node->position.turn() == WHITE) {
      node->position.play<WHITE>(move);
    } else {
      node->position.play<BLACK>(move);
    }

    Node child =
        Node(node, {}, node->position,
             policy[policyIndex(node->position, move)].item().toFloat());

    node->children.push_back(child);

    if (node->position.turn() == WHITE) {
      node->position.undo<WHITE>(move);
    } else {
      node->position.undo<BLACK>(move);
    }
  }
}

// applies the PUCT formula.
float puct(const Node &node) {
  return node.meanValue +
         (C_PUCT * node.policyEval *
          (sqrtf(node.parent->visitCount) / (1 + node.visitCount)));
}

void backup(Node *node, float val) {
  node->visitCount += 1;
  node->totalValue += val;
  node->meanValue = node->totalValue / node->visitCount;
}

// applies one MCTS simulation step to the specified node.
float simulate(Node *node, DNN &model) {
  if (isTerminal(node->position)) {
    float val = terminalValue(node->position);
    backup(node, val);
    return -val;
  } else if (node->children.size() == 0) {
    auto [value, policy] = model->forward(
        torch::unsqueeze(createState(constructHistory(node)), 0));

    float val = value.item<float>();
    expand(node, policy[0]);
    backup(node, val);
    return -val;
  }

  Node *selected = &node->children[0];

  for (Node &child : node->children) {
    if (puct(child) > puct(*selected)) {
      selected = &child;
    }
  }

  float val = simulate(selected, model);
  backup(selected, val);
  return -val;
}

// selects a node to play in mcts.
Node playMove(Node *root, DNN &model, float temperature) {
  for (int i = 0; i < 800; i++) {
    simulate(root, model);
  }

  float total = 0;
  for (Node &node : root->children) {
    total += pow(node.visitCount, 1.0 / temperature);
  }

  int i = rand() % static_cast<int>(total);
  float curr = 0;

  for (Node &node : root->children) {
    curr += pow(node.visitCount, 1.0 / temperature);
    if (curr < i) {
      return node;
    }
  }

  assert(false);
}