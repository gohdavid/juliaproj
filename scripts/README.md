Executable scripts and demos. Run from the repo root with `julia --project=.`
unless noted.

- `simulate_rouse_raw.jl`: generate raw, uncentered Rouse-chain trajectories
  from `configs/experiments/*.yaml`. Supports parallel worker runs and explicit
  `save.write_hdf5` / `save.write_jls` output switches.
- `rouse_score_analytics.jl`: load Rouse `.h5` or `.jls` outputs, combine
  workers, and make video, potential, or score diagnostics from
  `configs/plots/*.yaml`.
- `makie_config.jl`: shared CairoMakie plotting defaults/helpers.
- `polymer_langevin_makie.jl`: standalone polymer Langevin visualization demo.
- `spring_demo.jl`, `chignolin_demo.jl`, `extract_chignolin.py`: older toy or
  molecular demos retained for reference.
- `*.ipynb`: exploratory notebooks; prefer scripts/configs for reproducible
  runs.

Generated data should go under `outputs/`, `runs/`, or configured paths, not in
this folder.
