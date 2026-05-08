Run experiments from the repository root:

```sh
julia --project=. src/run_experiment.jl configs/nbody_mlp.yaml
julia --project=. src/run_experiment.jl configs/nbody_egnn.yaml
julia --project=. src/run_experiment.jl configs/nbody_flow_matching.yaml
julia --project=. src/run_experiment.jl configs/nbody_diffusion.yaml
julia --project=. src/run_experiment.jl configs/nbody_polymer_egnn.yaml
julia --project=. src/run_experiment.jl configs/nbody_polymer_flow_matching.yaml
julia --project=. src/run_experiment.jl configs/nbody_polymer_diffusion.yaml
```

The n-body configs expose the same knobs that were hard-coded in the notebooks:
data source, atom count, model type, hidden dimensions, trace estimator, ODE
tolerances, optimizer settings, sample count, output root, and seed. Configured
outputs are written under `<output.dir>/<experiment name>/<config-hash>`,
where the hash is computed from the non-output config fields.
Derived plot artifacts are written under the referenced data directory as
`<output.dir>/<experiment name>/<data-config-hash>/figures/<plot-config-hash>`.

`nbody_flow_matching.yaml` runs a center-of-mass-normalized E(n)-equivariant
conditional flow-matching baseline. It reports a pairwise-distance MAE metric
against the training distribution and writes the same `result.jls` /
`samples_vs_train.png` artifacts as the CNF configs.

`nbody_diffusion.yaml` runs a center-of-mass-normalized E(n)-equivariant DDPM
baseline with the same result artifacts and pairwise-distance MAE metric.

The `nbody_polymer_*.yaml` configs replace the fixed static toy with a 2D
uniform-temperature polymer Langevin dataset. The polymer is a linear chain with
adjacent harmonic backbone springs, independent Brownian noise per bead and
coordinate, and center-of-mass-aligned saved conformations. No correlated-noise
or correlation-matrix model is used.

The same configs can also train from precomputed Rouse HDF5 snapshots by using
`data.kind: rouse_hdf5`. The loader reads all matching `traj` datasets into RAM,
concatenates frames across files, and samples `data.n_samples` equilibrium
conformations from the combined ensemble. Use `data.source_dir` plus
`data.source_pattern` for seed-split outputs such as
`rouse_analysis_seed*.h5`, or provide `data.source_path` / `data.source_paths`
for explicit files.
