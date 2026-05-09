Raw Rouse/polymer trajectory-generation configs for
`scripts/simulate_rouse_raw.jl`.

- `rouse_base.yaml`: long 16-worker baseline Rouse run used by polymer model
  configs.
- `rouse_video.yaml`: short trajectory for visual checks.
- `rouse_*nonideal*.yaml`, `rouse_confinement.yaml`,
  `rouse_excluded_volume.yaml`, `rouse_lennard_jones.yaml`: variants with
  nonideal-force terms enabled or staged.

Output policy:

- `save.write_hdf5`: write streaming HDF5 trajectory files.
- `save.write_jls`: write serialized Julia payloads.
- Current Rouse configs are HDF5-only to avoid duplicating large trajectories.

Parallel runs derive worker seeds from `seed`, `parallel.workers`, and
`parallel.seed_stride`; worker files are named with `_seedN`.
