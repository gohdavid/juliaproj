#!/usr/bin/env python3
"""Run a solvated Chignolin simulation in OpenMM on one GPU.

Defaults target a long single-GPU production run:
- 10 microseconds of production dynamics
- checkpoints every 10 nanoseconds
- coordinates every 50 nanoseconds
- hydrogen mass repartitioning with a 4 fs timestep

Example:
    python simulate_chignolin.py

If no input PDB is provided, the script downloads PDB 1UAO from the RCSB.
"""

import argparse
import pathlib
import sys
import urllib.request

from openmm import LangevinMiddleIntegrator, Platform, unit
from openmm.app import (
    CheckpointReporter,
    DCDReporter,
    ForceField,
    HBonds,
    Modeller,
    PDBFile,
    PME,
    Simulation,
    StateDataReporter,
)


DEFAULT_PDB_ID = "1UAO"
DEFAULT_TIMESTEP_FS = 4.0
FS_PER_NS = 1_000_000.0
NS_PER_US = 1000.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-pdb",
        type=pathlib.Path,
        default=None,
        help="Local PDB file. If omitted, the script downloads --pdb-id.",
    )
    parser.add_argument(
        "--pdb-id",
        default=DEFAULT_PDB_ID,
        help=f"RCSB PDB ID to download when --input-pdb is omitted (default: {DEFAULT_PDB_ID}).",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("outputs/chignolin_10us"),
        help="Directory for trajectories, logs, and checkpoints.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=300.0,
        help="Simulation temperature in kelvin.",
    )
    parser.add_argument(
        "--friction",
        type=float,
        default=1.0,
        help="Langevin friction in 1/ps.",
    )
    parser.add_argument(
        "--timestep-fs",
        type=float,
        default=DEFAULT_TIMESTEP_FS,
        help="Integrator timestep in femtoseconds. 4 fs assumes hydrogen mass repartitioning.",
    )
    parser.add_argument(
        "--hydrogen-mass-amu",
        type=float,
        default=4.0,
        help="Hydrogen mass in amu used for repartitioning.",
    )
    parser.add_argument(
        "--padding-nm",
        type=float,
        default=1.0,
        help="Solvent padding in nm around the protein.",
    )
    parser.add_argument(
        "--ionic-strength-molar",
        type=float,
        default=0.15,
        help="Salt concentration used by addSolvent.",
    )
    parser.add_argument(
        "--equilibration-ns",
        type=float,
        default=1.0,
        help="Equilibration time in ns before production.",
    )
    parser.add_argument(
        "--production-us",
        type=float,
        default=10.0,
        help="Production length in microseconds.",
    )
    parser.add_argument(
        "--checkpoint-interval-ns",
        type=float,
        default=10.0,
        help="Interval in ns for checkpoint writes.",
    )
    parser.add_argument(
        "--coordinate-interval-ns",
        type=float,
        default=1.0,
        help="Interval in ns for DCD frames.",
    )
    parser.add_argument(
        "--log-interval-ns",
        type=float,
        default=1.0,
        help="Interval in ns for scalar logging to state.log.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for the integrator.",
    )
    parser.add_argument(
        "--cuda-device-index",
        default="0",
        help="CUDA device index. Use 0 for one H200.",
    )
    return parser.parse_args()


def ns_to_steps(time_ns, timestep_fs):
    return int(round(time_ns * FS_PER_NS / timestep_fs))


def us_to_steps(time_us, timestep_fs):
    return ns_to_steps(time_us * NS_PER_US, timestep_fs)


def ensure_pdb(input_pdb, pdb_id, output_dir):
    if input_pdb is not None:
        if not input_pdb.exists():
            raise FileNotFoundError(f"Input PDB not found: {input_pdb}")
        return input_pdb

    output_dir.mkdir(parents=True, exist_ok=True)
    pdb_path = output_dir / f"{pdb_id.lower()}.pdb"
    if pdb_path.exists():
        return pdb_path

    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    print(f"Downloading {pdb_id.upper()} from {url}", flush=True)
    urllib.request.urlretrieve(url, pdb_path)
    return pdb_path


def build_simulation(args, pdb_path):
    pdb = PDBFile(str(pdb_path))
    forcefield = ForceField("amber14-all.xml", "amber14/tip3p.xml")

    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(forcefield, pH=7.0)
    modeller.addSolvent(
        forcefield,
        model="tip3p",
        padding=args.padding_nm * unit.nanometer,
        ionicStrength=args.ionic_strength_molar * unit.molar,
    )

    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=HBonds,
        rigidWater=True,
        removeCMMotion=False,
        hydrogenMass=args.hydrogen_mass_amu * unit.amu,
        ewaldErrorTolerance=5.0e-4,
    )

    integrator = LangevinMiddleIntegrator(
        args.temperature * unit.kelvin,
        args.friction / unit.picosecond,
        args.timestep_fs * unit.femtoseconds,
    )
    integrator.setRandomNumberSeed(args.seed)
    integrator.setConstraintTolerance(1.0e-5)

    platform = Platform.getPlatformByName("CUDA")
    properties = {
        "DeviceIndex": args.cuda_device_index,
        "Precision": "mixed",
        "UseBlockingSync": "false",
        "DeterministicForces": "false",
    }

    simulation = Simulation(modeller.topology, system, integrator, platform, properties)
    simulation.context.setPositions(modeller.positions)
    return simulation, modeller


def configure_reporters(
    simulation,
    output_dir,
    checkpoint_interval_steps,
    coordinate_interval_steps,
    log_interval_steps,
    total_sim_steps,
    append=False,
):
    simulation.reporters.append(
        DCDReporter(str(output_dir / "trajectory.dcd"), coordinate_interval_steps, append=append)
    )
    simulation.reporters.append(
        CheckpointReporter(str(output_dir / "state.chk"), checkpoint_interval_steps, writeState=False)
    )
    simulation.reporters.append(
        StateDataReporter(
            str(output_dir / "state.log"),
            log_interval_steps,
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            density=True,
            speed=True,
            progress=True,
            remainingTime=True,
            totalSteps=total_sim_steps,
            separator="\t",
            append=append,
        )
    )


def run_in_chunks(simulation, total_steps, chunk_steps):
    completed = 0
    while completed < total_steps:
        step_count = min(chunk_steps, total_steps - completed)
        simulation.step(step_count)
        completed += step_count


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    equilibration_steps = ns_to_steps(args.equilibration_ns, args.timestep_fs)
    production_steps = us_to_steps(args.production_us, args.timestep_fs)
    checkpoint_interval_steps = ns_to_steps(args.checkpoint_interval_ns, args.timestep_fs)
    coordinate_interval_steps = ns_to_steps(args.coordinate_interval_ns, args.timestep_fs)
    log_interval_steps = ns_to_steps(args.log_interval_ns, args.timestep_fs)
    total_sim_steps = equilibration_steps + production_steps

    if checkpoint_interval_steps <= 0 or coordinate_interval_steps <= 0 or log_interval_steps <= 0:
        raise ValueError("Save and log intervals must map to at least one MD step.")
    if production_steps <= 0:
        raise ValueError("Production length must map to at least one MD step.")

    pdb_path = ensure_pdb(args.input_pdb, args.pdb_id, output_dir)
    print(f"Using PDB: {pdb_path}", flush=True)
    print(
        "Configured production: "
        f"{args.production_us:.3f} us = {production_steps} steps, "
        f"checkpoint interval {args.checkpoint_interval_ns:.3f} ns = {checkpoint_interval_steps} steps, "
        f"coordinate interval {args.coordinate_interval_ns:.3f} ns = {coordinate_interval_steps} steps, "
        f"log interval {args.log_interval_ns:.3f} ns = {log_interval_steps} steps",
        flush=True,
    )

    checkpoint_path = output_dir / "state.chk"
    solvated_pdb_path = output_dir / "solvated.pdb"

    simulation, modeller = build_simulation(args, pdb_path)
    atom_count = modeller.topology.getNumAtoms()
    print(f"System atom count after solvation: {atom_count}", flush=True)

    if checkpoint_path.exists():
        # ── Restart from checkpoint ──────────────────────────────────────────
        print(f"Checkpoint found: {checkpoint_path}", flush=True)
        simulation.loadCheckpoint(str(checkpoint_path))
        current_step = simulation.currentStep
        completed_production = current_step - equilibration_steps
        remaining_steps = production_steps - completed_production
        print(
            f"Resuming: {current_step} total steps done, "
            f"{completed_production} production steps done, "
            f"{remaining_steps} production steps remaining.",
            flush=True,
        )
        if remaining_steps <= 0:
            print("Production already complete.", flush=True)
            return 0

        configure_reporters(
            simulation,
            output_dir,
            checkpoint_interval_steps=checkpoint_interval_steps,
            coordinate_interval_steps=coordinate_interval_steps,
            log_interval_steps=log_interval_steps,
            total_sim_steps=total_sim_steps,
            append=True,
        )
        run_in_chunks(simulation, remaining_steps, checkpoint_interval_steps)

    else:
        # ── Fresh start ──────────────────────────────────────────────────────
        print("Minimizing...", flush=True)
        simulation.minimizeEnergy()

        print(f"Initializing velocities at {args.temperature:.1f} K", flush=True)
        simulation.context.setVelocitiesToTemperature(args.temperature * unit.kelvin, args.seed)

        # Write solvated topology PDB for downstream analysis tools
        if not solvated_pdb_path.exists():
            positions = simulation.context.getState(getPositions=True).getPositions()
            with open(solvated_pdb_path, "w", encoding="ascii") as fh:
                PDBFile.writeFile(modeller.topology, positions, fh)
            print(f"Solvated topology written to: {solvated_pdb_path}", flush=True)

        print(f"Equilibrating for {args.equilibration_ns:.3f} ns ({equilibration_steps} steps)", flush=True)
        run_in_chunks(simulation, equilibration_steps, min(log_interval_steps, equilibration_steps))

        configure_reporters(
            simulation,
            output_dir,
            checkpoint_interval_steps=checkpoint_interval_steps,
            coordinate_interval_steps=coordinate_interval_steps,
            log_interval_steps=log_interval_steps,
            total_sim_steps=total_sim_steps,
            append=False,
        )

        print(
            f"Running production for {args.production_us:.3f} us on CUDA device {args.cuda_device_index}",
            flush=True,
        )
        run_in_chunks(simulation, production_steps, checkpoint_interval_steps)

    positions = simulation.context.getState(getPositions=True).getPositions()
    with open(output_dir / "final.pdb", "w", encoding="ascii") as handle:
        PDBFile.writeFile(simulation.topology, positions, handle)

    print(f"Finished. Outputs written to: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
