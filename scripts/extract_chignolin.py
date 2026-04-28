#!/usr/bin/env python3
"""Extract pairwise Cα distances from the Chignolin DCD trajectory.

Pairwise distances are rotation- and translation-invariant, so no RMSD
alignment is needed.  A log transform makes the distribution roughly Gaussian
and keeps all values positive, which is better suited to a Gaussian-base CNF.

Outputs
-------
ca_distances.csv   — shape (n_frames, n_pairs), log(Angstrom)
                     n_pairs = n_ca*(n_ca-1)/2  =  45 for 10-residue Chignolin
ca_pair_labels.csv — one row per pair: "i,j,resname_i,resname_j"

Usage
-----
    python scripts/extract_chignolin.py
    python scripts/extract_chignolin.py --stride 10 --output-dir outputs/chignolin_10us
"""

import argparse
import pathlib
import sys

import numpy as np

try:
    import MDAnalysis as mda
except ImportError:
    sys.exit(
        "MDAnalysis not found.  Install with:\n"
        "  conda install -c conda-forge mdanalysis"
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("outputs/chignolin_10us"),
        help="Directory containing trajectory.dcd and solvated.pdb.",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=10,
        help=(
            "Use every Nth frame (default: 10).  "
            "At 1 ns DCD intervals, stride=10 gives one frame per 10 ns, "
            "which is roughly the Chignolin conformational decorrelation time."
        ),
    )
    return p.parse_args()


def pairwise_log_distances(positions):
    """Return log pairwise distances for upper triangle, shape (n_pairs,)."""
    n = positions.shape[0]
    diffs = positions[:, None, :] - positions[None, :, :]   # (n, n, 3)
    dist2 = np.sum(diffs ** 2, axis=-1)                      # (n, n)
    # Upper triangle indices (i < j)
    i_idx, j_idx = np.triu_indices(n, k=1)
    dists = np.sqrt(dist2[i_idx, j_idx])                     # (n_pairs,)
    return np.log(dists).astype(np.float32), i_idx, j_idx


def main():
    args = parse_args()
    out_dir = args.output_dir.resolve()

    topology   = out_dir / "solvated.pdb"
    trajectory = out_dir / "trajectory.dcd"
    out_csv    = out_dir / "ca_distances.csv"
    out_labels = out_dir / "ca_pair_labels.csv"

    if not topology.exists():
        sys.exit(
            f"Topology not found: {topology}\n"
            "Re-run simulate_chignolin.py from scratch to generate solvated.pdb."
        )
    if not trajectory.exists():
        sys.exit(f"Trajectory not found: {trajectory}")

    print(f"Loading: {topology} + {trajectory}", flush=True)
    u = mda.Universe(str(topology), str(trajectory))
    n_frames_total = len(u.trajectory)
    print(f"  {n_frames_total} frames, {u.atoms.n_atoms} atoms", flush=True)

    ca = u.select_atoms("protein and name CA")
    n_ca = ca.n_atoms
    n_pairs = n_ca * (n_ca - 1) // 2
    print(f"  Cα atoms: {n_ca}  →  {n_pairs} pairwise distances", flush=True)

    # Residue names for labelling
    resnames = [f"{r.resname}{r.resid}" for r in ca.residues]

    # Extract distances
    frame_indices = range(0, n_frames_total, args.stride)
    n_out = len(frame_indices)
    distances = np.empty((n_out, n_pairs), dtype=np.float32)

    i_idx = j_idx = None
    for out_i, fi in enumerate(frame_indices):
        u.trajectory[fi]
        log_d, i_idx, j_idx = pairwise_log_distances(ca.positions)
        distances[out_i] = log_d
        if (out_i + 1) % 200 == 0 or out_i == 0:
            print(f"  {out_i + 1}/{n_out} frames processed", flush=True)

    print(f"\nExtracted {distances.shape[0]} frames × {distances.shape[1]} log-distances", flush=True)

    # Save distances
    np.savetxt(str(out_csv), distances, delimiter=",", fmt="%.6f")
    print(f"Saved: {out_csv}", flush=True)

    # Save pair labels
    with open(out_labels, "w") as fh:
        fh.write("i,j,res_i,res_j\n")
        for i, j in zip(i_idx, j_idx):
            fh.write(f"{i},{j},{resnames[i]},{resnames[j]}\n")
    print(f"Saved: {out_labels}", flush=True)

    # Stats
    print("\nLog-distance stats:", flush=True)
    print(f"  mean: {distances.mean():.3f}  (exp = {np.exp(distances.mean()):.2f} Å)", flush=True)
    print(f"  std:  {distances.std():.3f}", flush=True)
    print(f"  min:  {distances.min():.3f}  (exp = {np.exp(distances.min()):.2f} Å)", flush=True)
    print(f"  max:  {distances.max():.3f}  (exp = {np.exp(distances.max()):.2f} Å)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
