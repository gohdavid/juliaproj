Model-training configs consumed by `src/run_experiment.jl`.

Naming convention:

- `cnf_mlp`: non-equivariant CNF MLP.
- `cnf_egnn`: equivariant CNF.
- `diffusion_egnn`: equivariant diffusion/score model.
- `flow_matching_egnn`: equivariant conditional flow-matching model.
- `nbody_polymer_*`: same model family trained on Rouse HDF5 snapshots instead
  of the static toy distribution.

Important fields:

- `experiment.family`: dispatch key used by `src/experiments.jl`.
- `experiment.name`: output-name component under `runs/`.
- `model.name`: human-readable model label using the naming convention above.
- `model.equivariant`: explicit boolean for paper/table bookkeeping.
- `diffusion.objective`: `score_matching` trains a time-dependent score network
  with denoising score matching; `denoising` trains the DDPM noise-prediction
  objective.
- `data.source_dir` / `data.source_pattern`: HDF5 source selection for polymer
  configs.
- `training.log_every`: epoch interval for training progress logs; `0` disables
  epoch logs.
- `training.checkpoint_every`: epoch interval for parameter checkpoints; `0`
  disables checkpointing. Checkpoints are written to
  `<output-dir>/checkpoints/checkpoint_epoch_*.jls` by default.

Run example:

```sh
julia --project=. src/run_experiment.jl configs/model/nbody_diffusion_egnn.yaml
```
