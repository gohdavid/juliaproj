#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-runs/nbody_polymer_hairpin_lj_cnf_egnn_3layer_rouse_hdf5/fa05e3687f52/checkpoints}"
SAMPLES_ROOT="${SAMPLES_ROOT:-runs/nbody_polymer_hairpin_lj_cnf_egnn_3layer_rouse_hdf5/fa05e3687f52/structural_checkpoint_eval}"
MAX_SAMPLES="${MAX_SAMPLES:-128}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MIN_EPOCH="${MIN_EPOCH:-2}"

extra_args=()
if [[ -n "${MAX_EPOCH:-}" ]]; then
  extra_args+=(--max-epoch "$MAX_EPOCH")
fi
if [[ "${FORCE:-0}" == "1" ]]; then
  extra_args+=(--force)
fi

julia --project=. scripts/eval_force_energy_checkpoints.jl \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --samples-root "$SAMPLES_ROOT" \
  --max-samples "$MAX_SAMPLES" \
  --batch-size "$BATCH_SIZE" \
  --min-epoch "$MIN_EPOCH" \
  "${extra_args[@]}"
