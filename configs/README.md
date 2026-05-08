Configuration files are split by workflow:

- `model/`: model-training configs for `src/run_experiment.jl`.
- `experiments/`: raw Rouse/polymer trajectory-generation configs for
  `scripts/simulate_rouse_raw.jl`.
- `plots/`: Rouse analysis/plot configs for `scripts/rouse_score_analytics.jl`.

Run model experiments from the repository root:

```sh
julia --project=. src/run_experiment.jl configs/model/nbody_cnf_mlp.yaml
julia --project=. src/run_experiment.jl configs/model/nbody_cnf_egnn.yaml
julia --project=. src/run_experiment.jl configs/model/nbody_flow_matching_egnn.yaml
julia --project=. src/run_experiment.jl configs/model/nbody_diffusion_egnn.yaml
julia --project=. src/run_experiment.jl configs/model/nbody_polymer_cnf_egnn.yaml
julia --project=. src/run_experiment.jl configs/model/nbody_polymer_flow_matching_egnn.yaml
julia --project=. src/run_experiment.jl configs/model/nbody_polymer_diffusion_egnn.yaml
```

The n-body configs expose the same knobs that were hard-coded in the notebooks:
data source, atom count, model type, hidden dimensions, trace estimator, ODE
tolerances, optimizer settings, sample count, output root, and seed. Configured
outputs are written under `<output.dir>/<experiment name>/<config-hash>`,
where the hash is computed from the non-output config fields.
Derived plot artifacts are written under the referenced data directory as
`<output.dir>/<experiment name>/<data-config-hash>/figures/<plot-config-hash>`.

`configs/model/nbody_flow_matching_egnn.yaml` runs a center-of-mass-normalized
E(n)-equivariant conditional flow-matching baseline. It reports a
pairwise-distance MAE metric against the training distribution and writes the
same `result.jls` / `samples_vs_train.png` artifacts as the CNF configs.

`configs/model/nbody_diffusion_egnn.yaml` runs a center-of-mass-normalized
E(n)-equivariant DDPM baseline with the same result artifacts and
pairwise-distance MAE metric.

The `configs/model/nbody_polymer_*.yaml` configs replace the fixed static toy
with a 2D uniform-temperature polymer Langevin dataset. The polymer is a linear
chain with adjacent harmonic backbone springs, independent Brownian noise per
bead and coordinate, and center-of-mass-aligned saved conformations. No
correlated-noise or correlation-matrix model is used.

The same configs can also train from precomputed Rouse HDF5 snapshots by using
`data.kind: rouse_hdf5`. The loader reads all matching `traj` datasets into RAM,
concatenates frames across files, and samples `data.n_samples` equilibrium
conformations from the combined ensemble. Use `data.source_dir` plus
`data.source_pattern` for seed-split outputs such as
`outputs/rouse_analysis/<run-hash>/trajectories` with
`rouse_analysis_seed*.h5`, or provide `data.source_path` /
`data.source_paths` for explicit files.
