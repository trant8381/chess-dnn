#!/usr/bin/env bash
# GPU saturation probe: run self-play at several PARALLEL_GAMES values (via
# CLI knobs, no source patching), sampling GPU utilization (nvidia-smi dmon),
# the train process CPU%, and the predictor's per-wave batch sizes
# (PREDICTOR_VERBOSE=1 + adaptive target_batch reporting).
#
# Usage: bash tests/run_gpu_probe.sh [PG1 PG2 ...]   (default: 32 64 128)
#
# Raw outputs and a summary table land in /tmp/probe_out/ (accumulates across
# invocations; rm -rf it to start fresh).
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=/tmp/probe_out
mkdir -p "$OUT"

PGS=${*:-"32 64 128"}
echo "nproc=$(nproc) gpus=$(nvidia-smi --query-gpu=count --format=csv,noheader)"

for PG in $PGS; do
  echo "== PG=$PG: running =="
  START=$(date +%s.%N)
  nvidia-smi dmon -s u -d 1 > "$OUT/gpu_${PG}.csv" &
  DMON=$!
  PREDICTOR_VERBOSE=1 ./build/train --total-games 96 --par-games "$PG" \
    --min-samples 64 > "$OUT/train_${PG}.log" 2>&1 &
  TPID=$!
  (
    while kill -0 "$TPID" 2>/dev/null; do
      ps -o %cpu= -p "$TPID" | tr -d ' ' >> "$OUT/cpu_${PG}.txt"
      sleep 2
    done
  ) &
  CPUP=$!
  set +e
  wait "$TPID"
  RC=$?
  set -e
  END=$(date +%s.%N)
  kill "$DMON" "$CPUP" 2>/dev/null || true
  wait "$DMON" "$CPUP" 2>/dev/null || true

  WALL=$(python3 -c "print('%.1f' % ($END - $START))")
  grep '^wave' "$OUT/train_${PG}.log" > "$OUT/waves_${PG}.txt" || true
  PRED=$(grep '^# predictor' "$OUT/train_${PG}.log" | tail -1)
  RUNE=$(grep '^# run end' "$OUT/train_${PG}.log" | tail -1)
  HDR=$(grep '^# chess-dnn' "$OUT/train_${PG}.log" | tail -1)
  GUTIL=$(awk 'NR>2 {v=$2+0; if(v>=0){s+=v; n++; if(v>mx)mx=v}} END {if(n) printf "avg=%.1f max=%.0f", s/n, mx; else print "n/a"}' "$OUT/gpu_${PG}.csv")
  GCORES=$(awk '{s+=$1; n++} END {if(n) printf "avg=%.1f", s/n; else print "n/a"}' "$OUT/cpu_${PG}.txt")
  HIST=$(awk '{f=$2; sub("n=", "", f); n=f+0} n<=8{a++} n<=31{b++} n<=63{c++} n<=127{d++} n<=255{e++} n<=511{fk++} n<=1023{g++} n<=2047{h++} END {printf "<=8:%d 9-31:%d 32-63:%d 64-127:%d 128-255:%d 256-511:%d 512-1023:%d >1023:%d total=%d", a, b-a, c-b, d-c, e-d, fk-e, g-fk, h-g, NR}' "$OUT/waves_${PG}.txt")
  {
    echo "===== PG=$PG (rc=$RC wall=${WALL}s) ====="
    echo "gpu_util_sm ${GUTIL}"
    echo "train_cpu_pct ${GCORES}"
    echo "waves $HIST"
    echo "header $HDR"
    echo "predictor $PRED"
    echo "runend $RUNE"
    echo ""
  } >> "$OUT/summary.txt"
  echo "--- PG=$PG rc=$RC wall=${WALL}s ---"
done

echo "========== SUMMARY =========="
cat "$OUT/summary.txt"
