Raw Rouse/polymer trajectory-generation configs for
`scripts/simulate_rouse_raw.jl`.

- `rouse_base.yaml`: long 16-worker baseline Rouse run and the source of truth
  for shared simulation settings and default nonideal-force parameters.
- `rouse_confinement.yaml`, `rouse_excluded_volume.yaml`,
  `rouse_lennard_jones.yaml`, `rouse_nonideal_additive.yaml`,
  `rouse_hairpin_lj.yaml`, and `rouse_ring_bond.yaml`: inherit from
  `rouse_base.yaml` and override only seed/log/output fields plus the knobs
  that enable their intended interactions. This keeps LJ, excluded-volume,
  confinement, solver, and save settings comparable across variants.
- `rouse_*_video.yaml`: inherit from the matching analysis config and override
  only short-run save settings, video output/log names, and
  `parallel.workers: 1`.
- `rouse_hairpin_lj.yaml`: additionally uses hairpin initialization and enables
  the inherited native-contact LJ term.
- `rouse_ring_bond.yaml`: additionally uses ring initialization and enables the
  inherited bonded end-to-end harmonic closure.

Output policy:

- `save.write_hdf5`: write streaming HDF5 trajectory files.
- `save.write_jls`: write serialized Julia payloads.
- Current Rouse configs are HDF5-only to avoid duplicating large trajectories.

Parallel runs derive worker seeds from `seed`, `parallel.workers`, and
`parallel.seed_stride`; worker files are named with `_seedN`.
