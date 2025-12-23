#include "constants.h"
#include "move_gen.h"
#include <ATen/core/TensorBody.h>
#include <cstdint>
#include <torch/torch.h>

using namespace Midnight;

torch::Tensor createState(const std::array<Position, HISTORY_BOARDS> &boards) {
  torch::Tensor boardState = torch::zeros({INPUT_PLANES, 8, 8});
  for (int i = 0; i < HISTORY_BOARDS; i++) {
    for (int pieceType = 0; pieceType < 6; pieceType++) {
      for (int color = 0; color < 2; color++) {
        uint64_t pieceBoard = boards[i].pieces[pieceType + color * 8];

        while (pieceBoard != 0) {
          int index = __builtin_ctzll(pieceBoard);
          boardState[i * 14 + color * 6 + pieceType][index / 8][index % 8] = 1;

          pieceBoard ^= 1ULL << index;
        }
      }
    }
    if (boards[i].has_repetition(Position::TWO_FOLD)) {
      boardState[i * 14 + 12] = torch::ones({8, 8});
    }
    if (boards[i].has_repetition(Position::THREE_FOLD)) {
      boardState[i * 14 + 13] = torch::ones({8, 8});
    }
  }

  boardState[14 * HISTORY_BOARDS] =
      boards[0].turn() == WHITE ? torch::zeros({8, 8}) : torch::ones({8, 8});
  boardState[14 * HISTORY_BOARDS + 1] = torch::full({8, 8}, boards[0].moves());
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
  boardState[14 * HISTORY_BOARDS + 6] = torch::full({8, 8}, boards[0].fifty_move_rule());

  return boardState;
}