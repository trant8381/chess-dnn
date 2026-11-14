# Tests

Small self-contained test executables, built alongside `train`/`play` (see
CMakeLists.txt). Each exits nonzero on the first failure; run with `cpu` or
`gpu` to force a device (default: GPU 0 when CUDA is available, else CPU).

| Test              | What it verifies                                             |
|-------------------|--------------------------------------------------------------|
| `test_movegen`    | Perft node counts (start, kiwipete, pos3, pos4), make/unmake round trips, fen parse/emit, stalemate and mate detection |
| `test_state_cpu`  | CPU `createState` 63-plane encoder against the bitboard/FEN truth (pieces, repetition, turn, move/fifty clocks, castling, history depth) |
| `test_state_fast` | GPU `createStateFast` kernels are bit-identical to the CPU encoder (single nodes and a 70-node batch) |
| `test_predictor`  | Predictor queue: eval delivery & shapes, values match a direct forward, publishWeights changes results, 8-thread contention, stop() drains |
| `test_mcts`       | End-to-end `getNextMove` through the predictor: legal temp-0/temp-1 selections, terminal roots, 4-ply pruned self-play |
| `run_smoke.sh`    | Integration: patches constants to a mini run, builds & runs `train` (self-play + optimizer + checkpoint), then an engine `play` round; restores constants even on failure |

Examples:

    ./build/test_movegen
    ./build/test_state_cpu
    ./build/test_predictor gpu
    ./build/test_mcts gpu
    ./build/test_state_fast         # CUDA builds only
    bash tests/run_smoke.sh

## Runtime knobs (no recompilation)

`train` accepts frequently-tuned settings on the command line:

    ./build/train [--total-games N] [--par-games N] [--simulations N]
                  [--min-samples N] [--train-batch N] [--replay-capacity N]
                  [--lr 1e-3]

Model-size/structural parameters stay compile-time in `src/include/constants.h`.
The predictor also self-corrects its max batch size while running (adaptive
batching, see `PREDICTOR_ADAPT_*` in `src/include/predictor.h`); watch it via
`PREDICTOR_VERBOSE=1` (per-wave lines) and `target_batch=` in the `# predictor`
summary line.
