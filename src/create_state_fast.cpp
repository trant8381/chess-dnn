#include "create_state_fast.h"
#include "move_gen.h"
#include "arange.cuh"
#include "batch_and_ne.cuh"
#include "sq.cuh"
#include <c10/core/ScalarType.h>
#include <torch/torch.h>
#include <vector>

// creates an array of the history boards.
NNInputBatch constructHistoryFast(std::vector<Node *> nodes) {
  NNInputBatch input;
  Histories &histories = input.histories;
  std::vector<std::array<float, 7>> &scalars = input.scalars;

  for (Node *node : nodes) {
    std::array<uint64_t, HISTORY_BOARDS * 14> history = {0};
    const Midnight::Position &board = node->position;
    const Node *current = node;

    auto createBoards = [&](const Midnight::Position &board, const int &i) {
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
         static_cast<float>(board.king_and_oo_rook_not_moved<Midnight::WHITE>()),
         static_cast<float>(board.king_and_ooo_rook_not_moved<Midnight::WHITE>()),
         static_cast<float>(board.king_and_oo_rook_not_moved<Midnight::BLACK>()),
         static_cast<float>(board.king_and_ooo_rook_not_moved<Midnight::BLACK>())});
  }

  return input;
}

torch::Tensor createStateFast(const std::vector<Node *> &nodes,
                              const torch::Device &device) {
  NNInputBatch input = constructHistoryFast(nodes);
  const long B = nodes.size();

  torch::Tensor batch =
      torch::from_blob((void *)input.histories.data(), {B, HISTORY_BOARDS * 14},
                       torch::TensorOptions().dtype(torch::kUInt64))
          .to(device);
  torch::Tensor mask =
      torch::empty(64, torch::TensorOptions().dtype(torch::kUInt64)).to(device);
  torch::Tensor binPlanes =
      torch::empty({B, 14, 8, 8},
                   torch::TensorOptions().dtype(torch::kFloat).device(device));
  
  uint64_t* maskPtr = mask.data_ptr<uint64_t>();
  uint64_t* batchPtr = batch.data_ptr<uint64_t>();
  float* binPlanesPtr = binPlanes.data_ptr<float>();

  arange(64, maskPtr);
  sq(64, maskPtr);
  batch_and_ne(1, maskPtr, batchPtr, binPlanesPtr);

  torch::Tensor scalar_vals =
      torch::from_blob(input.scalars.data(), {B, 7},
                       torch::TensorOptions().dtype(torch::kFloat))
          .to(device);

  torch::Tensor scalar_planes =
      scalar_vals.view({B, 7, 1, 1}).expand({B, 7, 8, 8});

  return torch::cat({binPlanes, scalar_planes}, 1);
}