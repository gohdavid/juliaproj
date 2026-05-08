#!/usr/bin/env julia

"""
Simulate and save one raw 2D Rouse-chain trajectory from a YAML config.

The saved coordinates are raw uncentered SDE states. No center-of-mass
correction is applied to the stored trajectory. This is intentional: the free
Rouse chain has a diffusing center of mass, while its internal conformations
are governed by the harmonic Gaussian-chain potential.

Model:

    dX_t = b(X_t) dt + sqrt(2D) dW_t

with independent Brownian noise for every bead coordinate and adjacent Rouse
spring drift

    b_i(X) = k_over_xi * (X_{i+1} - 2X_i + X_{i-1}).

The slowest Rouse relaxation time is

    tau_R = 1 / (k_over_xi * lambda_1),
    lambda_1 = 2 * (1 - cos(pi / N)).

Usage:

    julia --project=. scripts/simulate_rouse_raw.jl configs/rouse_video.yaml
    julia --project=. scripts/simulate_rouse_raw.jl configs/rouse_analysis.yaml
"""

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "src", "BoltzFlow.jl"))

using .BoltzFlow
using Logging
using ProgressLogging: ProgressLevel
using Random
using Serialization
using Printf
using StochasticDiffEq
using TerminalLoggers

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))

function project_path(path::AbstractString)
    return isabspath(path) ? path : joinpath(PROJECT_ROOT, path)
end

function rouse_relaxation_time(n::Int, k_over_xi::Real)
    lambda1 = 2.0f0 * (1.0f0 - cos(Float32(pi) / Float32(n)))
    return 1.0f0 / (Float32(k_over_xi) * lambda1)
end

function solution_to_traj(sol, dim::Int, n_beads::Int)
    traj = zeros(Float32, dim, n_beads, length(sol.u))
    for i in eachindex(sol.u)
        traj[:, :, i] = reshape(sol.u[i], dim, n_beads)
    end
    return traj
end

function build_save_times(until_tau::Real, interval_tau::Real, tau_r::Real)
    n_intervals = round(Int, Float64(until_tau) / Float64(interval_tau))
    tau_grid = Float32.(0:n_intervals) .* Float32(interval_tau)
    if abs(Float32(until_tau) - tau_grid[end]) > 100eps(Float32(until_tau))
        error("save.until_tau must be an integer multiple of save.interval_tau")
    end
    return tau_grid .* Float32(tau_r), tau_grid
end

function build_save_times(burn_in_tau::Real, until_tau::Real, interval_tau::Real,
                          tau_r::Real)
    save_times, save_tau = build_save_times(until_tau, interval_tau, tau_r)
    return save_times .+ Float32(burn_in_tau) .* Float32(tau_r), save_tau
end

function main()
    length(ARGS) == 1 || error("Usage: julia --project=. scripts/simulate_rouse_raw.jl <config.yaml>")
    cfg_path = project_path(ARGS[1])
    raw_cfg = BoltzFlow.load_yaml_config(cfg_path)

    seed = BoltzFlow.cfgint(raw_cfg, "seed", 11)
    rng = Xoshiro(seed)

    dim = BoltzFlow.cfgint(raw_cfg, "rouse.dim", 2)
    n_beads = BoltzFlow.cfgint(raw_cfg, "rouse.n_beads", 32)
    diffusion = BoltzFlow.cfgfloat32(raw_cfg, "rouse.diffusion", 1.0f0)
    bond_length = BoltzFlow.cfgfloat32(raw_cfg, "rouse.bond_length", 1.0f0)
    k_over_xi = BoltzFlow.cfgfloat32(raw_cfg, "rouse.k_over_xi",
                                     3.0f0 * diffusion / bond_length^2)
    dt = BoltzFlow.cfgfloat32(raw_cfg, "solver.dt", 0.01f0)
    burn_in_tau = BoltzFlow.cfgfloat32(raw_cfg, "save.burn_in_tau", 0.0f0)
    until_tau = BoltzFlow.cfgfloat32(raw_cfg, "save.until_tau", 5.0f0)
    interval_tau = BoltzFlow.cfgfloat32(raw_cfg, "save.interval_tau", 0.05f0)
    output_path = project_path(String(BoltzFlow.cfgget(raw_cfg, "output.path",
                                                       "outputs/rouse_raw/rouse.jls")))

    tau_r = rouse_relaxation_time(n_beads, k_over_xi)
    save_times, save_tau = build_save_times(burn_in_tau, until_tau, interval_tau, tau_r)
    total_steps = ceil(Int, save_times[end] / dt)

    sim_cfg = BoltzFlow.NBodyDataConfig(
        kind=:polymer_langevin,
        dim=dim,
        n_atoms=n_beads,
        n_samples=0,
        burn_in=0,
        total_steps=total_steps,
        dt=dt,
        physics_params=Float32[diffusion, bond_length, k_over_xi],
    )

    prob = BoltzFlow.polymer_langevin_sde_problem(rng, sim_cfg)
    logger = TerminalLogger(stderr, ProgressLevel; always_flush=true)
    sol = with_logger(logger) do
        solve(prob, EM(); dt=dt, saveat=save_times, adaptive=false,
              progress=true, progress_steps=10_000)
    end
    traj = solution_to_traj(sol, dim, n_beads)

    payload = Dict(
        "config_path" => cfg_path,
        "config" => Dict(
            "seed" => seed,
            "dim" => dim,
            "n_beads" => n_beads,
            "diffusion" => diffusion,
            "bond_length" => bond_length,
            "k_over_xi" => k_over_xi,
            "dt" => dt,
            "tau_r" => tau_r,
            "burn_in_tau" => burn_in_tau,
            "until_tau" => until_tau,
            "interval_tau" => interval_tau,
        ),
        "times" => Float32.(sol.t),
        "times_tau" => save_tau,
        "traj" => traj,
    )

    mkpath(dirname(output_path))
    serialize(output_path, payload)

    println("Saved raw Rouse trajectory: $output_path")
    @printf "config: %s\n" cfg_path
    @printf "tau_R: %.4f\n" tau_r
    @printf "burn-in: %.1f tau_R\n" burn_in_tau
    @printf "frames: %d every %.3f tau_R through %.1f tau_R after burn-in\n" size(traj, 3) interval_tau until_tau
end

main()
