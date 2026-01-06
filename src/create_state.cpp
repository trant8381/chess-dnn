#include "create_state.h"
#include "constants.h"
#include "mcts.h"
#include <ATen/ops/arange.h>
#include <ATen/ops/bitwise_right_shift.h>
#include <ATen/ops/cat.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/TensorOptions.h>
#include <torch/types.h>

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

// creates the input planes to be put into DNN.
// possibly need to normalize some of these features
torch::Tensor createState(const std::array<Position, HISTORY_BOARDS> &boards,
                          const torch::Device &device) {
  // initialize the planes.
  torch::Tensor boardState = torch::zeros({INPUT_PLANES, 8, 8}).to(device, 0);

  for (int i = 0; i < HISTORY_BOARDS; i++) {
    for (int pieceType = 0; pieceType < 6; pieceType++) {
      for (int color = 0; color < 2; color++) {
        uint64_t pieceBoard = boards[i].pieces[pieceType + color * 8];

        // loop to pop every bit from bitboard.
        while (pieceBoard != 0) {
          int index = __builtin_ctzll(pieceBoard);
          boardState[i * 14 + color * 6 + pieceType][index / 8][index % 8] = 1;

          pieceBoard ^= 1ULL << index;
        }
      }
    }
    // repetition boards.
    if (boards[i].has_repetition(Position::TWO_FOLD)) {
      boardState[i * 14 + 12] = torch::ones({8, 8});
    }
    if (boards[i].has_repetition(Position::THREE_FOLD)) {
      boardState[i * 14 + 13] = torch::ones({8, 8});
    }
  }

  // situational boards.

  // current turn.
  boardState[14 * HISTORY_BOARDS] =
      boards[0].turn() == WHITE ? torch::zeros({8, 8}) : torch::ones({8, 8});

  // halfmove count.
  boardState[14 * HISTORY_BOARDS + 1] =
      torch::full({8, 8}, boards[0].moves() / 100.0f);

  // castling.
  boardState[14 * HISTORY_BOARDS + 2] =
      boards[0].king_and_oo_rook_not_moved<WHITE>() ? torch::ones({8, 8})
                                                    : torch::zeros({8, 8});
  boardState[14 * HISTORY_BOARDS + 3] =
      boards[0].king_and_ooo_rook_not_moved<WHITE>() ? torch::ones({8, 8})
                                                     : torch::zeros({8, 8});
  boardState[14 * HISTORY_BOARDS + 4] =
      boards[0].king_and_oo_rook_not_moved<BLACK>() ? torch::ones({8, 8})
                                                    : torch::zeros({8, 8});
  boardState[14 * HISTORY_BOARDS + 5] =
      boards[0].king_and_ooo_rook_not_moved<BLACK>() ? torch::ones({8, 8})
                                                     : torch::zeros({8, 8});

  // fifty move rule. fifty_move_rule() returns the move count where there
  // hasn't been a pawn push or capture.
  boardState[14 * HISTORY_BOARDS + 6] =
      torch::full({8, 8}, boards[0].fifty_move_rule() / 100.0f);

  return boardState;
}

// creates an array of the history boards.
NNInputBatch constructHistoryFast(std::vector<Node *> nodes) {
  NNInputBatch input;
  Histories &histories = input.histories;
  std::vector<std::array<float, 7>> &scalars = input.scalars;

  for (Node *node : nodes) {
    std::array<uint64_t, HISTORY_BOARDS * 14> history = {0};
    const Position &board = node->position;
    const Node *current = node;

    auto createBoards = [&](const Position &board, const int &i) {
      for (int color = 0; color < 2; color++) {
        for (int pieceType = 0; pieceType < 6; pieceType++) {
          history[i * 14 + color * 6 + pieceType] =
              board.pieces[color * 8 + pieceType];
        }
      }
      history[i * 14 + 12] =
          board.has_repetition(Midnight::Position::TWO_FOLD) *
          0xffffffffffffffff;
      history[i * 14 + 13] =
          board.has_repetition(Midnight::Position::THREE_FOLD) *
          0xffffffffffffffff;
    };

    for (int i = 0; i < 4; i++) {
      if (current) {
        createBoards(current->position, i);
      } else {
        for (int j = 0; i + j < 4; j++) {
          createBoards(START_POS, i + j);
        }
        break;
      }
      current = current->parent;
    }

    histories.push_back(std::move(history));
    scalars.push_back(
        {board.fifty_move_rule() / 100.0f, board.moves() / 100.0f,
         static_cast<float>(board.turn()),
         static_cast<float>(board.king_and_oo_rook_not_moved<WHITE>()),
         static_cast<float>(board.king_and_ooo_rook_not_moved<WHITE>()),
         static_cast<float>(board.king_and_oo_rook_not_moved<BLACK>()),
         static_cast<float>(board.king_and_ooo_rook_not_moved<BLACK>())});
  }

  return input;
}

torch::Tensor createStateFast(const std::vector<Node *> &nodes,
                              const torch::Device &device) {
  NNInputBatch input = constructHistoryFast(nodes);
  const long B = nodes.size();

  torch::Tensor bb =
      torch::from_blob((void *)input.histories.data(), {B, HISTORY_BOARDS * 14},
                       torch::TensorOptions().dtype(torch::kUInt64))
          .to(device)
          .unsqueeze(2);

  static torch::Tensor squares = torch::arange(
      64, torch::TensorOptions().dtype(torch::kInt64).device(device));

  torch::Tensor shifted = torch::bitwise_right_shift(bb, squares);
  torch::Tensor bits = torch::bitwise_and(shifted, torch::ones_like(shifted));

  torch::Tensor bin_planes =
      bits.to(torch::kFloat).view({B, HISTORY_BOARDS * 14, 8, 8});

  torch::Tensor scalar_vals =
      torch::from_blob(input.scalars.data(), {B, 7},
                       torch::TensorOptions().dtype(torch::kFloat))
          .to(device);

  torch::Tensor scalar_planes =
      scalar_vals.view({B, 7, 1, 1}).expand({B, 7, 8, 8});

  return torch::cat({bin_planes, scalar_planes}, 1);
}
