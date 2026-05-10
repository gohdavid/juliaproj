#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p logs/rouse_batch logs/rouse_video_batch logs/rouse_video_plots

analysis_configs=(
  configs/experiments/rouse_base.yaml
  configs/experiments/rouse_lennard_jones.yaml
  configs/experiments/rouse_excluded_volume.yaml
  configs/experiments/rouse_confinement.yaml
  configs/experiments/rouse_nonideal_additive.yaml
  configs/experiments/rouse_hairpin_lj.yaml
  configs/experiments/rouse_ring_bond.yaml
)

video_configs=(
  configs/experiments/rouse_base_video.yaml
  configs/experiments/rouse_lennard_jones_video.yaml
  configs/experiments/rouse_excluded_volume_video.yaml
  configs/experiments/rouse_confinement_video.yaml
  configs/experiments/rouse_nonideal_additive_video.yaml
  configs/experiments/rouse_hairpin_lj_video.yaml
  configs/experiments/rouse_ring_bond_video.yaml
)

plot_configs=(
  configs/plots/rouse_base_video.yaml
  configs/plots/rouse_lennard_jones_video.yaml
  configs/plots/rouse_excluded_volume_video.yaml
  configs/plots/rouse_confinement_video.yaml
  configs/plots/rouse_nonideal_additive_video.yaml
  configs/plots/rouse_hairpin_lj_video.yaml
  configs/plots/rouse_ring_bond_video.yaml
)

run_batched() {
  local label="$1"
  local log_root="$2"
  local batch_size="$3"
  shift 3
  local configs=("$@")
  local failures=0
  local batch=()

  run_one_batch() {
    local pids=()
    local cfg base log
    for cfg in "${batch[@]}"; do
      base="$(basename "$cfg" .yaml)"
      log="${log_root}/${base}_parent.log"
      echo "[$(date --iso-8601=seconds)] starting ${label}: $cfg -> $log"
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

  local cfg
  for cfg in "${configs[@]}"; do
    batch+=("$cfg")
    if (( ${#batch[@]} == batch_size )); then
      run_one_batch
      batch=()
    fi
  done

  if (( ${#batch[@]} > 0 )); then
    run_one_batch
  fi

  if (( failures > 0 )); then
    echo "${failures} ${label} parent process(es) failed" >&2
    return 1
  fi
}

run_video_plots() {
  local failures=0
  local pids=()
  local cfg base log
  for cfg in "${plot_configs[@]}"; do
    base="$(basename "$cfg" .yaml)"
    log="logs/rouse_video_plots/${base}.log"
    echo "[$(date --iso-8601=seconds)] starting video plot: $cfg -> $log"
    JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      julia --project=. scripts/rouse_score_analytics.jl "$cfg" >"$log" 2>&1 &
    pids+=("$!")
  done

  local pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failures=$((failures + 1))
    fi
  done

  if (( failures > 0 )); then
    echo "${failures} video plot process(es) failed" >&2
    return 1
  fi
}

run_batched "analysis simulation" logs/rouse_batch 4 "${analysis_configs[@]}"
run_batched "video simulation" logs/rouse_video_batch 7 "${video_configs[@]}"
run_video_plots

echo "All Rouse simulations, video simulations, and normalized video plots completed."
