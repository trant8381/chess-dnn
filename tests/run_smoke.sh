#!/usr/bin/env bash
# End-to-end smoke test: build, run a tiny CLI-driven training session, check
# the training log + checkpoint, play one engine move against it.
#
# All knobs are passed on the command line (see src/train.cpp --help), so this
# script never patches or touches sources.
set -euo pipefail
cd "$(dirname "$0")/.."

cleanup() {
  rm -f training.log checkpoint.pt
}
trap cleanup EXIT

echo "== building train =="
cmake --build build --target train -j"$(nproc)" >/dev/null

echo "== mini training run (CLI knobs) =="
timeout 1800 ./build/train --total-games 4 --par-games 4 \
  --min-samples 64 --train-batch 64 --simulations 80 >/dev/null 2>&1

echo "== assertions =="
test -f training.log || { echo "FAIL: training.log missing"; exit 1; }
grep -q '^# chess-dnn training run' training.log || { echo "FAIL: no header"; exit 1; }
test "$(grep -c '^step,policy_loss' training.log)" = "1" || { echo "FAIL: bad CSV header"; exit 1; }
ROWS=$(grep -c '^[0-9]\+,' training.log || true)
echo "data rows: $ROWS"
test "$ROWS" -ge 1 || { echo "FAIL: no training step rows"; exit 1; }
grep -q "^# run end:" training.log || { echo "FAIL: no run-end summary"; exit 1; }
test -f checkpoint.pt || { echo "FAIL: checkpoint.pt missing"; exit 1; }
echo "training.log OK ($(wc -l < training.log) lines)"

echo "== engine play round =="
printf 'e2e4\nquit\n' | ./build/play checkpoint.pt white 2>/dev/null |
  grep -q 'Engine plays' || { echo "FAIL: engine did not play"; exit 1; }
echo "engine play OK"

echo "SMOKE PASS"
