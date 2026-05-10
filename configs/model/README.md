Model-training configs consumed by `src/run_experiment.jl`.

Naming convention:

- `cnf_mlp`: non-equivariant CNF MLP.
- `cnf_egnn`: equivariant CNF.
- `cnf_egnn_1m`: equivariant CNF with about 1M parameters.
- `diffusion_egnn`: equivariant diffusion/score model.
- `flow_matching_egnn`: equivariant conditional flow-matching model.
- `nbody_polymer_*`: same model family trained on Rouse HDF5 snapshots instead
  of the static toy distribution.
- `nbody_polymer_hairpin_lj_cnf_egnn.yaml`: CNF config for the stable
  hairpin-contact Rouse ensemble.
Important fields:

- `experiment.family`: dispatch key used by `src/experiments.jl`.
- `experiment.name`: output-name component under `runs/`.
- `model.name`: human-readable model label using the naming convention above.
- `model.equivariant`: explicit boolean for paper/table bookkeeping.
- `diffusion_egnn` trains a VP-SDE time-dependent score network directly
  against `-eps / sigma(t)` with `sigma_square(t)` loss weighting.
- `data.source_dir` / `data.source_pattern`: HDF5 source selection for polymer
  configs.
- `training.log_every`: step interval for training progress logs; `0` disables
  progress logs.
- `training.checkpoint_every`: epoch interval for parameter checkpoints; default
  is every epoch. Set `0` to disable checkpointing. Checkpoints are written to
  `<output-dir>/checkpoints/checkpoint_epoch_*.jls` by default.

Run example:

```sh
julia --project=. src/run_experiment.jl configs/model/nbody_diffusion_egnn.yaml
```
