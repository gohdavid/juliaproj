#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p logs/d025_run2

configs=(
  configs/experiments/d025_run2/rouse_base.yaml
  configs/experiments/d025_run2/rouse_lennard_jones.yaml
  configs/experiments/d025_run2/rouse_excluded_volume.yaml
  configs/experiments/d025_run2/rouse_confinement.yaml
  configs/experiments/d025_run2/rouse_nonideal_additive.yaml
  configs/experiments/d025_run2/rouse_hairpin_lj.yaml
  configs/experiments/d025_run2/rouse_ring_bond.yaml
)

batch_size=4
failures=0

run_batch() {
  local -n batch_configs=$1
  local pids=()
  local cfg base log

  for cfg in "${batch_configs[@]}"; do
    base="$(basename "$cfg" .yaml)"
    log="logs/d025_run2/${base}_parent.log"
    echo "[$(date --iso-8601=seconds)] starting $cfg -> $log"
    JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      julia --project=. scripts/simulate_rouse_raw.jl "$cfg" >"$log" 2>&1 &
    pids+=("$!")
  done

  local pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failures=$((failures + 1))
    fi
  done
}

batch=()
for cfg in "${configs[@]}"; do
  batch+=("$cfg")
  if (( ${#batch[@]} == batch_size )); then
    run_batch batch
    batch=()
  fi
done

if (( ${#batch[@]} > 0 )); then
  run_batch batch
fi

if (( failures > 0 )); then
  echo "$failures simulation parent process(es) failed" >&2
  exit 1
fi

echo "All d025 run2 simulations completed."
