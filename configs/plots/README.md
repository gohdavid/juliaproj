Plot/analysis configs consumed by `scripts/rouse_score_analytics.jl`.

- `rouse_base_video.yaml`: make baseline trajectory animation frames/GIF.
- `rouse_*_video.yaml`: make trajectory animation frames/GIF for matching
  experiment variants.
- `rouse_potential.yaml`: make analytic potential diagnostics.
- `rouse_score.yaml`: make analytic score-norm diagnostics.

These configs point back to an experiment config through `experiment.config`.
With `data.prefer_hdf5: true`, the analysis loads worker HDF5 files from the
experiment's hashed output directory. Plot outputs are written under that data
directory in `figures/<plot-config-hash>/`.
