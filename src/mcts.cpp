#include "mcts.h"
#include "constants.h"
#include "create_state.h"
#include "dnn.h"
#include "move_gen.h"
#include <cstdint>
#include <vector>

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

std::array<Position, HISTORY_BOARDS> constructHistory(Node *node) {
  std::array<Position, HISTORY_BOARDS> history;
  Node* currNode = node;
  for (int i = 0; i < HISTORY_BOARDS; i++) {
    history[i] = currNode->position;
    currNode = currNode->parent;
  }

  return history;
}

// wip
float simulate(Node* node, DNN &model) {
  if (isTerminal(node->position)) {
    return terminalValue(node->position);
  } else if (node->children.size() == 0) {
    auto [value, policy] = model->forward(createState(constructHistory(node)));

    return value.item().toFloat();
  }
}