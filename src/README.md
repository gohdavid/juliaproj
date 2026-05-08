Core Julia source for `BoltzFlow`.

- `BoltzFlow.jl`: module entry point; exports public APIs and includes the
  model/data/config/experiment files.
- `config.jl`: dependency-light YAML subset parser plus deterministic config
  hashes and output-directory helpers.
- `run_experiment.jl`: CLI wrapper for model configs:
  `julia --project=. src/run_experiment.jl <config.yaml>`.
- `experiments.jl`: dispatch from `experiment.family` to CNF, flow-matching, or
  diffusion training, sampling, plotting, and result serialization.
- `nbody_data.jl`: static toy data, polymer/Rouse simulation helpers, HDF5
  loading, centering, sampling, and analytic polymer scores.
- `nbody_cnf.jl`: CNF vector fields (`cnf_mlp`, `cnf_egnn`), likelihood loss,
  Adam training, and reverse sampling.
- `nbody_flow_matching.jl`: equivariant EGNN-style flow-matching baseline
  (`flow_matching_egnn`).
- `nbody_diffusion.jl`: equivariant EGNN-style diffusion/score baseline
  (`diffusion_egnn`).

Keep reusable logic here. Put one-off runs, plotting jobs, and data-generation
entry points in `scripts/`.
