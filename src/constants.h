#pragma once

#include "move_gen.h"
#include <cmath>
#include <cstdint>
#include <torch/torch.h>

constexpr int HISTORY_BOARDS = 4; // model's amount of history positions stored.
constexpr int INPUT_PLANES =
    HISTORY_BOARDS * 14 +
    7; // (6 white pieces + 6 black pieces + 2 repetitions) per history board. 7
       // situational planes.
constexpr int TRUNK_CHANNELS = 64; // channels per resnet block.
constexpr int TOWER_SIZE = 6;      // amount of resnet blocks.
constexpr float C_PUCT = 1.5f;     // PUCT constant for MCTS selection.
constexpr int SIMULATIONS = 200;   // amount of simulations for one move.
constexpr float FPU = -0.2f;       // temperature constant for move selection.
constexpr uint64_t TABLE_SIZE = 1ULL << 25; // size of transposition table.
constexpr float UNKNOWN = INFINITY;         // unknown value for batchPUCT.
constexpr float VL = 2; // virtual loss value for updating statistics.
constexpr float TEMPERATURE_DECAY = -0.42f; // the exponent for temperature.
constexpr int PARALLEL_GAMES =
    2; // the number of games to be run in parallel during data collection.
const Midnight::Position START_POS = Midnight::Position(Midnight::START_FEN);