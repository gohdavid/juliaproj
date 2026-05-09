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
    julia --project=. scripts/simulate_rouse_raw.jl configs/experiments/rouse_base.yaml
"""

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "src", "BoltzFlow.jl"))

using .BoltzFlow
using DiffEqCallbacks: PresetTimeCallback
using HDF5
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
    lambda1 = 2.0 * (1.0 - cos(pi / Float64(n)))
    return 1.0 / (Float64(k_over_xi) * lambda1)
end

function solution_to_traj(sol, dim::Int, n_beads::Int)
    traj = zeros(Float32, dim, n_beads, length(sol.u))
    for i in eachindex(sol.u)
        traj[:, :, i] = reshape(sol.u[i], dim, n_beads)
    end
    return traj
end

function frame_hdf5_path(output_path::AbstractString)
    root, ext = splitext(output_path)
    return isempty(ext) ? string(output_path, ".h5") : string(root, ".h5")
end

function output_paths(raw_cfg)
    legacy_output = BoltzFlow.cfgget(raw_cfg, "output.path", nothing)
    output_dir = BoltzFlow.cfgget(raw_cfg, "output.dir", nothing)

    if output_dir !== nothing
        out_dir = project_path(BoltzFlow.config_output_dir(raw_cfg; default_name="rouse_raw"))
        default_file = legacy_output === nothing ? "traj.jls" : basename(String(legacy_output))
        output_file = String(BoltzFlow.cfgget(raw_cfg, "output.file", default_file))
        output_path = joinpath(out_dir, output_file)

        legacy_hdf5 = BoltzFlow.cfgget(raw_cfg, "output.hdf5_path", nothing)
        default_hdf5 = legacy_hdf5 === nothing ? basename(frame_hdf5_path(output_path)) :
                       basename(String(legacy_hdf5))
        hdf5_file = String(BoltzFlow.cfgget(raw_cfg, "output.hdf5_file", default_hdf5))
        return output_path, joinpath(out_dir, hdf5_file)
    end

    output_path = project_path(String(BoltzFlow.cfgget(raw_cfg, "output.path",
                                                       "outputs/rouse_raw/traj.jls")))
    hdf5_path = project_path(String(BoltzFlow.cfgget(raw_cfg, "output.hdf5_path",
                                                     frame_hdf5_path(output_path))))
    return output_path, hdf5_path
end

function cfgset!(cfg::AbstractDict, path::AbstractString, value)
    cur = cfg
    parts = split(path, ".")
    for key in parts[1:end-1]
        if !(haskey(cur, key) && cur[key] isa AbstractDict)
            cur[key] = Dict{String,Any}()
        end
        cur = cur[key]
    end
    cur[parts[end]] = value
    return cfg
end

function seed_filename(filename::AbstractString, seed::Int)
    root, ext = splitext(filename)
    return string(root, "_seed", seed, ext)
end

function worker_config(raw_cfg, worker_id::Int)
    cfg = deepcopy(raw_cfg)
    seed = worker_seed(raw_cfg, worker_id)
    output_file = String(BoltzFlow.cfgget(cfg, "output.file", "traj.jls"))
    hdf5_file = String(BoltzFlow.cfgget(cfg, "output.hdf5_file",
                                        seed_filename(frame_hdf5_path(output_file), seed)))
    cfgset!(cfg, "output.file", seed_filename(output_file, seed))
    cfgset!(cfg, "output.hdf5_file", seed_filename(hdf5_file, seed))
    return cfg
end

function worker_seed(raw_cfg, worker_id::Int)
    seed = BoltzFlow.cfgint(raw_cfg, "seed", 11)
    stride = BoltzFlow.cfgint(raw_cfg, "parallel.seed_stride", 1)
    return seed + (worker_id - 1) * stride
end

function parse_worker_id(args)
    worker_id = nothing
    positional = String[]
    for arg in args
        if startswith(arg, "--worker-id=")
            worker_id = parse(Int, split(arg, "="; limit=2)[2])
        else
            push!(positional, arg)
        end
    end
    return positional, worker_id
end

function run_parallel_parent(cfg_path::AbstractString, raw_cfg, n_workers::Int)
    script_path = abspath(@__FILE__)
    log_dir = project_path(String(BoltzFlow.cfgget(raw_cfg, "parallel.log_dir", "logs")))
    child_threads = string(BoltzFlow.cfgint(raw_cfg, "parallel.julia_num_threads", 1))
    mkpath(log_dir)

    procs = []
    for worker_id in 1:n_workers
        seed = worker_seed(raw_cfg, worker_id)
        experiment_name = String(BoltzFlow.cfgget(raw_cfg, "output.experiment_name", "rouse_raw"))
        log_path = joinpath(log_dir, "$(experiment_name)_seed$(seed).log")
        cmd = `$(Base.julia_cmd()) --project=$(PROJECT_ROOT) $script_path $cfg_path --worker-id=$worker_id`
        env = copy(ENV)
        env["JULIA_NUM_THREADS"] = child_threads
        env["OPENBLAS_NUM_THREADS"] = "1"
        open(log_path, "w") do io
            proc = run(pipeline(setenv(cmd, env); stdout=io, stderr=io); wait=false)
            push!(procs, proc)
            println("started worker $worker_id seed=$seed pid=$(getpid(proc)) log=$log_path")
        end
    end

    failed = Int[]
    for (worker_id, proc) in enumerate(procs)
        wait(proc)
        if !success(proc)
            push!(failed, worker_id)
        end
    end
    isempty(failed) || error("Rouse workers failed: $(join(failed, ", "))")
    println("Completed $n_workers parallel Rouse workers.")
end

function build_save_times(until_tau::Real, interval_tau::Real, tau_r::Real)
    n_intervals = round(Int, Float64(until_tau) / Float64(interval_tau))
    tau_grid = Float64.(0:n_intervals) .* Float64(interval_tau)
    tolerance = 100eps(max(abs(Float64(until_tau)), abs(tau_grid[end]), 1.0))
    if abs(Float64(until_tau) - tau_grid[end]) > tolerance
        error("save.until_tau must be an integer multiple of save.interval_tau")
    end
    return tau_grid .* Float64(tau_r), tau_grid
end

function build_save_times(burn_in_tau::Real, until_tau::Real, interval_tau::Real,
                          tau_r::Real)
    save_times, save_tau = build_save_times(until_tau, interval_tau, tau_r)
    return save_times .+ Float64(burn_in_tau) .* Float64(tau_r), save_tau
end

function sde_solver(name)
    normalized = lowercase(replace(String(name), r"[-_\s]" => ""))
    if normalized == "em"
        return EM()
    elseif normalized == "eulerheun"
        return EulerHeun()
    elseif normalized == "rkmil"
        return RKMil()
    elseif normalized == "sosra"
        return SOSRA()
    end
    error("Unsupported solver.algorithm=$(repr(name)); supported values are EM, EulerHeun, RKMil, SOSRA")
end

function cfgflag(cfg, path::AbstractString, default::Bool=false)
    return Bool(BoltzFlow.cfgget(cfg, path, default))
end

function build_nonideal_params(raw_cfg)
    lj_enabled = cfgflag(raw_cfg, "nonideal.lennard_jones.enabled", false)
    lj_epsilon = BoltzFlow.cfgfloat(raw_cfg, "nonideal.lennard_jones.epsilon", 0.0)
    lj_sigma = BoltzFlow.cfgfloat(raw_cfg, "nonideal.lennard_jones.sigma", 1.0)
    lj_softening = BoltzFlow.cfgfloat(raw_cfg, "nonideal.lennard_jones.softening", 0.0)
    lj_cutoff = BoltzFlow.cfgfloat(raw_cfg, "nonideal.lennard_jones.cutoff", 0.0)
    lj_exclude_bonded = cfgflag(raw_cfg, "nonideal.lennard_jones.exclude_bonded", true)
    lj_shift = cfgflag(raw_cfg, "nonideal.lennard_jones.shift", true)

    ev_enabled = cfgflag(raw_cfg, "nonideal.excluded_volume.enabled", false)
    ev_epsilon = BoltzFlow.cfgfloat(raw_cfg, "nonideal.excluded_volume.epsilon", 0.0)
    ev_sigma = BoltzFlow.cfgfloat(raw_cfg, "nonideal.excluded_volume.sigma", 1.0)
    ev_softening = BoltzFlow.cfgfloat(raw_cfg, "nonideal.excluded_volume.softening", 0.0)
    ev_power = BoltzFlow.cfgfloat(raw_cfg, "nonideal.excluded_volume.power", 12.0)
    ev_cutoff = BoltzFlow.cfgfloat(raw_cfg, "nonideal.excluded_volume.cutoff", 0.0)
    ev_exclude_bonded = cfgflag(raw_cfg, "nonideal.excluded_volume.exclude_bonded", true)

    conf_enabled = cfgflag(raw_cfg, "nonideal.confinement.enabled", false)
    conf_strength = BoltzFlow.cfgfloat(raw_cfg, "nonideal.confinement.strength", 0.0)
    conf_centered = cfgflag(raw_cfg, "nonideal.confinement.centered", true)

    packed = Float64[
        lj_enabled ? 1.0 : 0.0,
        lj_epsilon,
        lj_sigma,
        lj_softening,
        lj_cutoff,
        lj_exclude_bonded ? 1.0 : 0.0,
        lj_shift ? 1.0 : 0.0,
        ev_enabled ? 1.0 : 0.0,
        ev_epsilon,
        ev_sigma,
        ev_softening,
        ev_power,
        ev_cutoff,
        ev_exclude_bonded ? 1.0 : 0.0,
        conf_enabled ? 1.0 : 0.0,
        conf_strength,
        conf_centered ? 1.0 : 0.0,
    ]
    config = Dict(
        "lennard_jones" => Dict(
            "enabled" => lj_enabled,
            "epsilon" => lj_epsilon,
            "sigma" => lj_sigma,
            "softening" => lj_softening,
            "cutoff" => lj_cutoff,
            "exclude_bonded" => lj_exclude_bonded,
            "shift" => lj_shift,
        ),
        "excluded_volume" => Dict(
            "enabled" => ev_enabled,
            "epsilon" => ev_epsilon,
            "sigma" => ev_sigma,
            "softening" => ev_softening,
            "power" => ev_power,
            "cutoff" => ev_cutoff,
            "exclude_bonded" => ev_exclude_bonded,
        ),
        "confinement" => Dict(
            "enabled" => conf_enabled,
            "strength" => conf_strength,
            "centered" => conf_centered,
        ),
    )
    return packed, config
end

function solve_streaming_hdf5(prob, solver, dt::Real, save_times, save_tau, dim::Int,
                              n_beads::Int, hdf5_path::AbstractString;
                              progress::Bool=true, progress_steps::Int=10_000)
    traj = zeros(Float32, dim, n_beads, length(save_times))
    times = zeros(Float64, length(save_times))
    next_frame = Ref(1)

    mkpath(dirname(hdf5_path))
    h5open(hdf5_path, "w") do h5
        attrs(h5)["format"] = "rouse_raw_hdf5_v1"
        attrs(h5)["dim"] = dim
        attrs(h5)["n_beads"] = n_beads
        attrs(h5)["n_frames"] = length(save_times)
        h5["traj"] = zeros(Float32, dim, n_beads, length(save_times))
        h5["times"] = zeros(Float64, length(save_times))
        h5["times_tau"] = Float64.(save_tau)
        h5["written"] = zeros(Int32, 1)
        flush(h5)

        function write_frame!(u, t)
            i = next_frame[]
            i > length(save_times) && return nothing
            frame = reshape(u, dim, n_beads)
            traj[:, :, i] .= frame
            times[i] = Float64(t)
            h5["traj"][:, :, i] = traj[:, :, i]
            h5["times"][i] = times[i]
            h5["written"][1] = Int32(i)
            flush(h5)
            next_frame[] = i + 1
            return nothing
        end

        if !isempty(save_times) && iszero(save_times[1])
            write_frame!(prob.u0, 0.0f0)
        end

        function affect!(integrator)
            write_frame!(integrator.u, integrator.t)
            return nothing
        end
        callback_times = typeof(dt).(save_times[next_frame[]:end])
        callback = PresetTimeCallback(callback_times, affect!;
                                      save_positions=(false, false))

        solve(prob, solver; dt=dt, adaptive=false, callback,
              save_everystep=false, save_start=false, save_end=false,
              progress, progress_steps)
    end

    next_frame[] == length(save_times) + 1 ||
        error("Only wrote $(next_frame[] - 1) of $(length(save_times)) requested frames")
    return traj, times
end

function main()
    positional, worker_id = parse_worker_id(ARGS)
    length(positional) == 1 ||
        error("Usage: julia --project=. scripts/simulate_rouse_raw.jl <config.yaml> [--worker-id=N]")
    cfg_path = project_path(positional[1])
    raw_cfg = BoltzFlow.load_yaml_config(cfg_path)

    n_workers = BoltzFlow.cfgint(raw_cfg, "parallel.workers", 1)
    if worker_id === nothing && n_workers > 1
        run_parallel_parent(cfg_path, raw_cfg, n_workers)
        return nothing
    elseif worker_id !== nothing
        1 <= worker_id <= n_workers || error("--worker-id=$worker_id outside 1:$n_workers")
        raw_cfg = worker_config(raw_cfg, worker_id)
    end

    seed = worker_id === nothing ? BoltzFlow.cfgint(raw_cfg, "seed", 11) :
           worker_seed(BoltzFlow.load_yaml_config(cfg_path), worker_id)
    rng = Xoshiro(seed)

    dim = BoltzFlow.cfgint(raw_cfg, "rouse.dim", 2)
    n_beads = BoltzFlow.cfgint(raw_cfg, "rouse.n_beads", 32)
    diffusion = BoltzFlow.cfgfloat(raw_cfg, "rouse.diffusion", 1.0)
    bond_length = BoltzFlow.cfgfloat(raw_cfg, "rouse.bond_length", 1.0)
    k_over_xi = BoltzFlow.cfgfloat(raw_cfg, "rouse.k_over_xi",
                                   3.0 * diffusion / bond_length^2)
    solver_algorithm = String(BoltzFlow.cfgget(raw_cfg, "solver.algorithm", "EM"))
    solver = sde_solver(solver_algorithm)
    dt = BoltzFlow.cfgfloat(raw_cfg, "solver.dt", 0.01)
    burn_in_tau = BoltzFlow.cfgfloat(raw_cfg, "save.burn_in_tau", 0.0)
    until_tau = BoltzFlow.cfgfloat(raw_cfg, "save.until_tau", 5.0)
    interval_tau = BoltzFlow.cfgfloat(raw_cfg, "save.interval_tau", 0.05)
    write_hdf5 = BoltzFlow.cfgbool(
        raw_cfg, "save.write_hdf5",
        BoltzFlow.cfgbool(raw_cfg, "save.stream_hdf5", true),
    )
    write_jls = BoltzFlow.cfgbool(raw_cfg, "save.write_jls", !write_hdf5)
    (write_hdf5 || write_jls) ||
        error("At least one of save.write_hdf5 or save.write_jls must be true")
    log_progress = BoltzFlow.cfgbool(raw_cfg, "logging.progress", true)
    progress_steps = BoltzFlow.cfgint(raw_cfg, "logging.progress_steps", 10_000)
    output_path, hdf5_path = output_paths(raw_cfg)
    nonideal_params, nonideal_config = build_nonideal_params(raw_cfg)

    tau_r = rouse_relaxation_time(n_beads, k_over_xi)
    save_times, save_tau = build_save_times(burn_in_tau, until_tau, interval_tau, tau_r)
    dt64 = Float64(dt)
    total_steps = ceil(Int, save_times[end] / dt64) + 2

    sim_cfg = BoltzFlow.NBodyDataConfig(
        kind=:polymer_langevin,
        dim=dim,
        n_atoms=n_beads,
        n_samples=0,
        burn_in=0,
        total_steps=total_steps,
        dt=dt,
        physics_params=vcat(Float64[diffusion, bond_length, k_over_xi], nonideal_params),
    )

    prob = BoltzFlow.polymer_langevin_sde_problem(rng, sim_cfg)
    logger = TerminalLogger(stderr, ProgressLevel; always_flush=true)
    traj, times = with_logger(logger) do
        if write_hdf5
            solve_streaming_hdf5(prob, solver, dt64, save_times, save_tau, dim, n_beads,
                                 hdf5_path; progress=log_progress,
                                 progress_steps=progress_steps)
        else
            sol = solve(prob, solver; dt=dt64, saveat=save_times, adaptive=false,
                        progress=log_progress, progress_steps=progress_steps)
            (solution_to_traj(sol, dim, n_beads), Float64.(sol.t))
        end
    end

    if write_jls
        payload = Dict(
            "config_path" => cfg_path,
            "config" => Dict(
                "seed" => seed,
                "dim" => dim,
                "n_beads" => n_beads,
                "diffusion" => diffusion,
                "bond_length" => bond_length,
                "k_over_xi" => k_over_xi,
                "solver_algorithm" => solver_algorithm,
                "dt" => dt,
                "tau_r" => tau_r,
                "burn_in_tau" => burn_in_tau,
                "until_tau" => until_tau,
                "interval_tau" => interval_tau,
                "hdf5_path" => write_hdf5 ? hdf5_path : nothing,
                "nonideal" => nonideal_config,
            ),
            "times" => times,
            "times_tau" => save_tau,
            "traj" => traj,
        )

        mkpath(dirname(output_path))
        serialize(output_path, payload)
        println("Saved raw Rouse trajectory: $output_path")
    end
    write_hdf5 && println("Saved HDF5 trajectory:     $hdf5_path")
    @printf "config: %s\n" cfg_path
    @printf "solver: %s\n" solver_algorithm
    @printf "tau_R: %.4f\n" tau_r
    @printf "burn-in: %.1f tau_R\n" burn_in_tau
    @printf "frames: %d every %.3f tau_R through %.1f tau_R after burn-in\n" size(traj, 3) interval_tau until_tau
    worker_id === nothing || @printf "worker: %d seed: %d\n" worker_id seed
end

main()
