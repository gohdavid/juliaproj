# D=0.25 Rouse/Polymer Second Run

This directory contains overlay configs for a second low-diffusion simulation
set. The original `D=1.0` configs under `configs/experiments/` are left
unchanged.

## Purpose

These runs lower the Rouse/polymer diffusion from `1.0` to `0.25` to create a
tighter, lower-temperature-like training distribution for CNF overfitting and
NLL-floor sanity checks.

With `k_over_xi: 3.0` unchanged, the effective stiffness ratio
`k_over_xi / diffusion` increases from `3` to `12`. This makes conformations
less noisy and narrows bond fluctuations, but it is a different target
distribution from the original `D=1.0` data.

## Output Separation

All configs use suffixed experiment names such as:

- `rouse_base_d025_run2`
- `rouse_lennard_jones_d025_run2`
- `rouse_excluded_volume_d025_run2`
- `rouse_confinement_d025_run2`
- `rouse_nonideal_additive_d025_run2`
- `rouse_hairpin_lj_d025_run2`
- `rouse_ring_bond_d025_run2`

The video configs similarly use names ending in `_video_d025_run2`.

This keeps `D=0.25` outputs separate from the existing `D=1.0` outputs under
`outputs/`. Stale files are intentionally not removed.

## Long Simulations

Each long config runs `16` workers. To run four configs at a time, use:

```sh
scripts/run_rouse_d025_run2_batch.sh
```

That script launches the seven long configs in batches of four parent
processes, giving up to `4 x 16 = 64` worker simulations concurrently.

Individual configs can also be run directly:

```sh
julia --project=. scripts/simulate_rouse_raw.jl configs/experiments/d025_run2/rouse_base.yaml
```

## Video Simulations

The `*_video.yaml` configs are short, one-worker runs through `5 tau_R` with
`0.01 tau_R` frame spacing. They inherit the same `D=0.25` setting and write to
separate `_video_d025_run2` output directories.

Example:

```sh
julia --project=. scripts/simulate_rouse_raw.jl configs/experiments/d025_run2/rouse_base_video.yaml
```

Matching plot configs live in `configs/plots/d025_run2/`.

Example:

```sh
julia --project=. scripts/rouse_score_analytics.jl configs/plots/d025_run2/rouse_base_video.yaml
```

## Notes

- These configs are overlays. Shared solver, save, nonideal-force, and output
  file settings still come from the original parent configs.
- Do not compare raw CNF NLL values against `D=1.0` runs as if they are the
  same target distribution.
- For overfitting checks, use these datasets to validate capacity and training
  behavior, then separately decide whether the final model should target
  `D=1.0` or `D=0.25`.
