#!/usr/bin/env julia

"""
Generate a Brownian Rouse/polymer trajectory using a trained CNF or diffusion
checkpoint as the score model.

The learned score is interpreted as

    score(x) = grad_x log p_model(x)

and converted into the overdamped Langevin drift/force with

    force(x) = D * score(x)
    dX_t = force(X_t) dt + sqrt(2D) dW_t.

The default `--dynamics overdamped` mode matches the existing polymer Langevin
data workflow. `--dynamics underdamped` uses Hamiltonian Langevin dynamics with
momentum and the same learned force.

By default `D` is read from `checkpoint["config"]["data"]["physics_params"][1]`.
Use `--diffusion` or `--polymer-config` to override/infer it from a matching
Rouse experiment config.

Usage:

  julia --project=. scripts/brownian_rouse_trajectory.jl --checkpoint runs/.../checkpoints/checkpoint_epoch_000060.jls
  julia --project=. scripts/brownian_rouse_trajectory.jl --checkpoint checkpoint.jls --polymer-config configs/experiments/rouse_base.yaml --steps 50000 --save-stride 100
"""

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CairoMakie
using Dates
using DiffEqCallbacks
using LinearAlgebra
using Printf
using Random
using Serialization
using Statistics
using StochasticDiffEq

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))

push!(LOAD_PATH, PROJECT_ROOT)
include(joinpath(PROJECT_ROOT, "src", "BoltzFlow.jl"))
using .BoltzFlow

function usage()
    println("""
    Usage:
      julia --project=. scripts/brownian_rouse_trajectory.jl --checkpoint CHECKPOINT.jls [options]

    Required:
      --checkpoint PATH          CNF or diffusion checkpoint .jls.

    Polymer / dynamics:
      --polymer-config PATH      Optional Rouse experiment YAML for D, k/xi, and dt defaults.
      --diffusion D              Override diffusion coefficient used to convert score to force.
      --dt DT                    Integration timestep. Default: polymer config solver.dt, else checkpoint data.dt, else 5e-4.
      --steps N                  Number of Euler-Maruyama steps. Default: 20000.
      --save-stride N            Save every N steps. Default: 100.
      --log-interval-steps N     Log solve progress every N steps. Default: 1. Use <=0 to disable.
      --score-clip VALUE         Clip per-frame score Frobenius norm. Default: 128. Use <=0 to disable.
      --diffusion-time T         Score time for diffusion checkpoints. Default: 1e-5.
      --dynamics MODE            overdamped or underdamped. Default: overdamped.
      --friction GAMMA           Underdamped friction. Default: 1.0.
      --mass MASS                Underdamped mass. Default: 1.0.
      --init {data,noise}        Initial condition. Default: data if reference data loads, else noise.

    Evaluation / output:
      --output-dir DIR           Default: <checkpoint-run>/brownian_rouse_trajectory/<timestamp>.
      --max-data-samples N       Limit ground-truth frames for metrics. Default: 4000.
      --metric-samples N         Limit sim/reference frames in distribution metrics. Default: 512.
      --metric-values N          Limit scalar distances/energies in 1D metrics. Default: 2000.
      --framerate N              GIF framerate. Default: 24.
      --no-gif                   Skip GIF rendering.
      --help                     Show this message.
    """)
end

function parse_args(args)
    opts = Dict{String,Any}(
        "steps" => 20_000,
        "save_stride" => 100,
        "log_interval_steps" => 1,
        "score_clip" => 128.0,
        "diffusion_time" => 1.0e-5,
        "max_data_samples" => 4000,
        "metric_samples" => 512,
        "metric_values" => 2000,
        "framerate" => 24,
        "gif" => true,
        "init" => "data",
        "dynamics" => "overdamped",
        "friction" => 1.0,
        "mass" => 1.0,
    )

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("--help", "-h")
            usage()
            exit(0)
        elseif arg == "--no-gif"
            opts["gif"] = false
            i += 1
        elseif arg in ("--checkpoint", "--polymer-config", "--output-dir",
                       "--init", "--dynamics")
            i < length(args) || error("Missing value for $arg")
            opts[replace(arg[3:end], "-" => "_")] = args[i + 1]
            i += 2
        elseif arg in ("--steps", "--save-stride", "--log-interval-steps",
                       "--max-data-samples", "--metric-samples",
                       "--metric-values", "--framerate")
            i < length(args) || error("Missing value for $arg")
            opts[replace(arg[3:end], "-" => "_")] = parse(Int, args[i + 1])
            i += 2
        elseif arg in ("--diffusion", "--dt", "--score-clip",
                       "--diffusion-time", "--friction", "--mass")
            i < length(args) || error("Missing value for $arg")
            opts[replace(arg[3:end], "-" => "_")] = parse(Float64, args[i + 1])
            i += 2
        else
            error("Unknown argument: $arg")
        end
    end

    haskey(opts, "checkpoint") || error("Set --checkpoint PATH")
    opts["init"] in ("data", "noise") || error("--init must be data or noise")
    opts["dynamics"] in ("overdamped", "underdamped") ||
        error("--dynamics must be overdamped or underdamped")
    return opts
end

project_path(path::AbstractString) =
    isabspath(path) ? normpath(path) : normpath(joinpath(PROJECT_ROOT, path))

function latest_output_dir(checkpoint_path::AbstractString)
    ckpt_dir = dirname(checkpoint_path)
    run_dir = basename(ckpt_dir) == "checkpoints" ? dirname(ckpt_dir) : ckpt_dir
    stamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    return joinpath(run_dir, "brownian_rouse_trajectory", stamp)
end

function rouse_relaxation_time(n::Int, k_over_xi::Real)
    lambda1 = 2.0 * (1.0 - cos(pi / Float64(n)))
    return 1.0 / (Float64(k_over_xi) * lambda1)
end

function experiment_physics(polymer_cfg::Union{Nothing,Dict{String,Any}})
    polymer_cfg === nothing && return nothing
    diffusion = BoltzFlow.cfgfloat(polymer_cfg, "rouse.diffusion", 1.0)
    bond_length = BoltzFlow.cfgfloat(polymer_cfg, "rouse.bond_length", 1.0)
    k_over_xi = BoltzFlow.cfgfloat(polymer_cfg, "rouse.k_over_xi",
                                   3.0 * diffusion / bond_length^2)
    dt = BoltzFlow.cfgfloat(polymer_cfg, "solver.dt", 5.0e-4)
    return (; diffusion, bond_length, k_over_xi, dt)
end

function checkpoint_physics(cfg::Dict{String,Any})
    p = Float64.(BoltzFlow.cfgget(cfg, "data.physics_params", Float64[]))
    diffusion = length(p) >= 1 ? p[1] : 1.0
    bond_length = length(p) >= 2 ? p[2] : 1.0
    k_over_xi = length(p) >= 3 ? p[3] : 3.0 * diffusion / bond_length^2
    dt = BoltzFlow.cfgfloat(cfg, "data.dt", 5.0e-4)
    return (; diffusion, bond_length, k_over_xi, dt)
end

function build_diffusion_context(cfg::Dict{String,Any}, device)
    data_cfg = BoltzFlow._data_config(cfg)
    n_layers = BoltzFlow.cfgint(cfg, "model.n_layers",
                                BoltzFlow.cfgint(cfg, "model.egnn_layers", 4))
    model = BoltzFlow.build_diffusion_model(
        dim=data_cfg.dim,
        n_atoms=data_cfg.n_atoms,
        hidden_dims=Int.(BoltzFlow.cfgget(cfg, "model.hidden_dims", [64, 64, 64])),
        n_layers=n_layers,
        node_embedding_dim=BoltzFlow.cfgint(cfg, "model.node_embedding_dim",
                                            min(16, data_cfg.n_atoms)),
        normalize_edge_features=BoltzFlow.cfgbool(
            cfg, "model.normalize_edge_features", true),
        seq_dist_scale=BoltzFlow.cfgfloat32(
            cfg, "model.seq_dist_scale", max(data_cfg.n_atoms - 1, 1)),
        radial_scale=BoltzFlow.cfgfloat32(
            cfg, "model.radial_scale", max(data_cfg.n_atoms, 1)),
        coordinate_update_scale=BoltzFlow.cfgfloat32(
            cfg, "model.coordinate_update_scale", 0.1f0),
    )
    sample_steps = BoltzFlow.cfgint(cfg, "sampling.steps",
                                    BoltzFlow.cfgint(cfg, "diffusion.steps", 1000))
    return BoltzFlow.NBodyDiffusionContext(model, device, sample_steps), data_cfg
end

function model_context_from_checkpoint(checkpoint)
    cfg = checkpoint["config"]
    family = BoltzFlow.cfgsymbol(cfg, "experiment.family", "nbody_cnf")
    device = BoltzFlow._device_from_config(cfg)
    if family == :nbody_cnf
        ctx, data_cfg = BoltzFlow._cnf_context_from_config(cfg, device)
        return (; family, ctx, data_cfg, device)
    elseif family in (:nbody_diffusion, :nbody_equivariant_diffusion)
        ctx, data_cfg = build_diffusion_context(cfg, device)
        return (; family, ctx, data_cfg, device)
    end
    error("Unsupported checkpoint family $family; expected nbody_cnf or nbody_diffusion")
end

function load_reference_data(cfg::Dict{String,Any}, rng::AbstractRNG, max_samples::Int)
    data_cfg = BoltzFlow._data_config(cfg)
    data = Float32.(BoltzFlow.generate_nbody_dataset(rng, data_cfg))
    if BoltzFlow.cfgbool(cfg, "data.center", true)
        data = BoltzFlow.center_positions(data)
    end
    if max_samples > 0 && size(data, 3) > max_samples
        idx = randperm(rng, size(data, 3))[1:max_samples]
        data = data[:, :, idx]
    end
    return data, data_cfg
end

function score_function(model_ctx, params, normalizer, diffusion_time::Real)
    cpu = BoltzFlow.DiffEqFlux.Lux.cpu_device()
    family = model_ctx.family
    ctx = model_ctx.ctx
    rng = Xoshiro(12345)
    return function (x::AbstractMatrix)
        batch = reshape(Float32.(x), size(x, 1), size(x, 2), 1)
        if family == :nbody_cnf
            score = cpu(BoltzFlow.normalized_cnf_logp_gradient(
                ctx, params, batch, normalizer; rng))
            return Float64.(score[:, :, 1])
        end

        x_norm = BoltzFlow.center_positions(BoltzFlow.apply_data_normalizer(
            batch, normalizer))
        pred = cpu(BoltzFlow.diffusion_logp_gradient(
            ctx, params, x_norm; time=Float32(diffusion_time)))
        if Bool(get(normalizer, "enabled", false))
            pred = pred ./ normalizer["scale"]
        end
        return Float64.(BoltzFlow.center_positions(pred)[:, :, 1])
    end
end

function clipped_score(score::AbstractMatrix, max_norm::Real)
    max_norm > 0 || return score
    nrm = norm(score)
    if isfinite(nrm) && nrm > max_norm
        return score .* (Float64(max_norm) / nrm)
    elseif !isfinite(nrm)
        return zero(score)
    end
    return score
end

function learned_score_sde_problem(x0, dim::Int, n_atoms::Int, diffusion::Real,
                                   dt::Real, steps::Int, score_fn;
                                   score_clip::Real=128.0,
                                   dynamics::AbstractString="overdamped",
                                   friction::Real=1.0,
                                   mass::Real=1.0,
                                   rng::AbstractRNG=Random.default_rng())
    drift = zeros(Float64, dim, n_atoms)
    tspan = (0.0, Float64(dt) * Float64(steps))

    if dynamics == "overdamped"
        function overdamped_f!(du, u, _p, _t)
            x = reshape(u, dim, n_atoms)
            score = clipped_score(score_fn(x), score_clip)
            drift .= Float64(diffusion) .* score
            reshape(du, dim, n_atoms) .= drift
            return nothing
        end

        sigma = sqrt(2.0 * Float64(diffusion))
        function overdamped_g!(du, _u, _p, _t)
            fill!(du, sigma)
            return nothing
        end

        return SDEProblem(overdamped_f!, overdamped_g!, vec(Float64.(x0)),
                          tspan, nothing)
    end

    mass64 = Float64(mass)
    gamma64 = Float64(friction)
    mass64 > 0 || error("--mass must be positive")
    gamma64 >= 0 || error("--friction must be nonnegative")
    v0 = sqrt(Float64(diffusion) / mass64) .* randn(rng, Float64, dim, n_atoms)
    state0 = vcat(vec(Float64.(x0)), vec(v0))
    n_x = dim * n_atoms

    function underdamped_f!(du, u, _p, _t)
        x = reshape(@view(u[1:n_x]), dim, n_atoms)
        v = reshape(@view(u[(n_x + 1):end]), dim, n_atoms)
        dx = reshape(@view(du[1:n_x]), dim, n_atoms)
        dv = reshape(@view(du[(n_x + 1):end]), dim, n_atoms)
        score = clipped_score(score_fn(x), score_clip)
        drift .= Float64(diffusion) .* score
        dx .= v
        dv .= drift ./ mass64 .- gamma64 .* v
        return nothing
    end

    sigma_v = sqrt(2.0 * gamma64 * Float64(diffusion) / mass64)
    function underdamped_g!(du, _u, _p, _t)
        du[1:n_x] .= 0.0
        du[(n_x + 1):end] .= sigma_v
        return nothing
    end

    return SDEProblem(underdamped_f!, underdamped_g!, state0, tspan, nothing)
end

function format_duration(seconds::Real)
    total = max(0, round(Int, seconds))
    hours = total ÷ 3600
    minutes = (total % 3600) ÷ 60
    secs = total % 60
    hours > 0 && return @sprintf("%dh%02dm%02ds", hours, minutes, secs)
    minutes > 0 && return @sprintf("%dm%02ds", minutes, secs)
    return @sprintf("%ds", secs)
end

function solve_progress_callback(steps::Int, dt::Real, log_interval_steps::Int,
                                 start_wall::Float64)
    log_interval_steps > 0 || return nothing
    interval_t = Float64(dt) * Float64(log_interval_steps)
    interval_t > 0 || return nothing
    last_step = Ref(-1)

    function log_progress(integrator)
        step = clamp(round(Int, integrator.t / Float64(dt)), 0, steps)
        step == last_step[] && return nothing
        last_step[] = step
        elapsed = time() - start_wall
        fraction = steps > 0 ? step / steps : 1.0
        eta = (step > 0 && fraction > 0) ?
              format_duration(elapsed * (1.0 / fraction - 1.0)) :
              "unknown"
        @info "solve progress" step steps percent=round(100.0 * fraction; digits=1) elapsed=format_duration(elapsed) eta
        flush(stderr)
        return nothing
    end

    return PeriodicCallback(log_progress, interval_t;
                            initial_affect=true, final_affect=true)
end

function solution_to_traj(sol, dim::Int, n_atoms::Int)
    traj = zeros(Float32, dim, n_atoms, length(sol.u))
    n_x = dim * n_atoms
    for i in eachindex(sol.u)
        traj[:, :, i] = Float32.(BoltzFlow.center_positions(
            reshape(@view(sol.u[i][1:n_x]), dim, n_atoms)))
    end
    return traj
end

function chain_points(x)
    return Point2f.(x[1, :], x[2, :])
end

function axis_limits(traj; pad=0.18f0)
    xmin, xmax = extrema(Float32.(traj[1, :, :]))
    ymin, ymax = extrema(Float32.(traj[2, :, :]))
    span = max(xmax - xmin, ymax - ymin, 1.0f0)
    xmid = (xmin + xmax) / 2.0f0
    ymid = (ymin + ymax) / 2.0f0
    half_width = (0.5f0 + pad) * span
    return (xmid - half_width, xmid + half_width),
           (ymid - half_width, ymid + half_width)
end

function draw_chain!(ax, pts, n_beads::Int; draw_ring_bond::Bool=false)
    lines!(ax, pts; color=:gray20, linewidth=4)
    if draw_ring_bond && n_beads > 2
        lines!(ax, Point2f[pts[end], pts[1]]; color=:gray55, linewidth=4)
    end
    scatter!(ax, pts; color=1:n_beads, colormap=:viridis,
             markersize=16, strokewidth=0)
end

function save_chain_frames(traj, times, frame_dir; draw_ring_bond::Bool=false)
    mkpath(frame_dir)
    xlims, ylims = axis_limits(traj)
    n_frames = size(traj, 3)
    n_beads = size(traj, 2)
    log_every = max(1, n_frames ÷ 10)
    @info "saving trajectory frames" frame_dir n_frames
    flush(stderr)
    for frame in 1:n_frames
        fig = Figure(size=(700, 700), backgroundcolor=:white)
        ax = Axis(fig[1, 1], aspect=DataAspect(), xlabel="x", ylabel="y")
        ax.xgridvisible[] = false
        ax.ygridvisible[] = false
        hidespines!(ax, :t, :r)
        xlims!(ax, xlims...)
        ylims!(ax, ylims...)
        draw_chain!(ax, chain_points(traj[:, :, frame]), n_beads; draw_ring_bond)
        text!(ax, xlims[1] + 0.04 * (xlims[2] - xlims[1]),
              ylims[2] - 0.06 * (ylims[2] - ylims[1]),
              text=@sprintf("t = %.3g", times[frame]), fontsize=24)
        save(joinpath(frame_dir, @sprintf("frame_%05d.png", frame)), fig)
        if frame == 1 || frame == n_frames || frame % log_every == 0
            @info "frame progress" frame n_frames percent=round(100.0 * frame / n_frames; digits=1)
            flush(stderr)
        end
    end
    return frame_dir
end

function record_gif(traj, times, gif_path; framerate::Int=24,
                    draw_ring_bond::Bool=false)
    xlims, ylims = axis_limits(traj)
    n_beads = size(traj, 2)
    frame_ids = collect(axes(traj, 3))
    fig = Figure(size=(700, 700), backgroundcolor=:white)
    ax = Axis(fig[1, 1], aspect=DataAspect(), xlabel="x", ylabel="y")
    ax.xgridvisible[] = false
    ax.ygridvisible[] = false
    hidespines!(ax, :t, :r)
    xlims!(ax, xlims...)
    ylims!(ax, ylims...)

    @info "rendering trajectory gif" gif_path n_frames=length(frame_ids) framerate
    flush(stderr)
    record(fig, gif_path, frame_ids; framerate) do frame
        empty!(ax)
        draw_chain!(ax, chain_points(traj[:, :, frame]), n_beads; draw_ring_bond)
        text!(ax, xlims[1] + 0.04 * (xlims[2] - xlims[1]),
              ylims[2] - 0.06 * (ylims[2] - ylims[1]),
              text=@sprintf("t = %.3g", times[frame]), fontsize=24)
    end
    @info "finished trajectory gif" gif_path
    flush(stderr)
    return gif_path
end

function polymer_potentials(samples, data_cfg)
    p = data_cfg.physics_params
    diffusion = length(p) >= 1 ? p[1] : 1.0
    bond_length = length(p) >= 2 ? p[2] : 1.0
    k_over_xi = length(p) >= 3 ? p[3] : 3.0 * diffusion / bond_length^2
    nonideal = BoltzFlow.polymer_nonideal_params(p)
    values = Vector{Float32}(undef, size(samples, 3))
    for i in axes(samples, 3)
        values[i] = Float32(BoltzFlow.polymer_langevin_potential(
            @view(samples[:, :, i]), diffusion, k_over_xi, nonideal))
    end
    return values
end

function upper_pairwise_features(samples)
    dists = BoltzFlow.pairwise_distances(samples)
    n_atoms = size(dists, 1)
    n_pairs = n_atoms * (n_atoms - 1) ÷ 2
    feats = Matrix{Float32}(undef, n_pairs, size(samples, 3))
    row = 1
    for j in 2:n_atoms, i in 1:(j - 1)
        feats[row, :] .= dists[i, j, :]
        row += 1
    end
    return feats
end

finite_values(x) = Float64.(filter(isfinite, vec(collect(x))))

function mean_pairwise_abs(x, y)
    (isempty(x) || isempty(y)) && return NaN
    total = 0.0
    for b in y, a in x
        total += abs(a - b)
    end
    return total / (length(x) * length(y))
end

function scalar_distribution_energy(x, y)
    (isempty(x) || isempty(y)) && return Float32(NaN)
    value = 2.0 * mean_pairwise_abs(x, y) -
            mean_pairwise_abs(x, x) -
            mean_pairwise_abs(y, y)
    return Float32(max(value, 0.0))
end

function median_pairwise_abs(values; max_points::Int=512)
    n = length(values)
    n_eval = min(n, max_points)
    n_eval >= 2 || return 1.0
    idx = round.(Int, range(1, n; length=n_eval))
    distances = Float64[]
    sizehint!(distances, n_eval * (n_eval - 1) ÷ 2)
    for b in 2:n_eval, a in 1:(b - 1)
        push!(distances, abs(values[idx[a]] - values[idx[b]]))
    end
    return max(median!(distances), eps(Float64))
end

function mean_rbf_kernel(x, y, sigma::Real)
    sigma2 = Float64(sigma)^2
    total = 0.0
    for b in y, a in x
        delta = a - b
        total += exp(-0.5 * delta * delta / sigma2)
    end
    return total / (length(x) * length(y))
end

function scalar_distribution_mmd(x, y)
    (isempty(x) || isempty(y)) && return Float32(NaN)
    base_sigma = median_pairwise_abs(vcat(x, y))
    sigmas = [0.5 * base_sigma, base_sigma, 2.0 * base_sigma, 4.0 * base_sigma]
    value = mean(mean_rbf_kernel(x, x, sigma) +
                 mean_rbf_kernel(y, y, sigma) -
                 2.0 * mean_rbf_kernel(x, y, sigma)
                 for sigma in sigmas)
    return Float32(max(value, 0.0))
end

function limit_frames(samples, max_samples::Int, rng::AbstractRNG)
    max_samples > 0 || return samples
    n = size(samples, 3)
    n <= max_samples && return samples
    idx = randperm(rng, n)[1:max_samples]
    return samples[:, :, idx]
end

function limit_values(values, max_values::Int, rng::AbstractRNG)
    vals = finite_values(values)
    max_values > 0 || return vals
    length(vals) <= max_values && return vals
    return vals[randperm(rng, length(vals))[1:max_values]]
end

function distribution_metrics(reference, simulated, data_cfg, rng::AbstractRNG;
                              metric_samples::Int=512,
                              metric_values::Int=2000)
    ref = limit_frames(reference, metric_samples, rng)
    sim = limit_frames(simulated, metric_samples, rng)

    ref_energy = polymer_potentials(ref, data_cfg)
    sim_energy = polymer_potentials(sim, data_cfg)
    ref_pw = limit_values(upper_pairwise_features(ref), metric_values, rng)
    sim_pw = limit_values(upper_pairwise_features(sim), metric_values, rng)

    metrics = Dict{String,Any}(
        "n_reference_frames" => size(ref, 3),
        "n_simulated_frames" => size(sim, 3),
        "n_pairwise_distance_values" => min(length(ref_pw), length(sim_pw)),
        "pairwise_distance_mae" => BoltzFlow.pairwise_distance_mae(ref, sim),
        "pairwise_distance_scalar_mmd" => scalar_distribution_mmd(ref_pw, sim_pw),
        "pairwise_distance_scalar_energy_distance" => scalar_distribution_energy(ref_pw, sim_pw),
        "polymer_energy_mmd" => scalar_distribution_mmd(finite_values(ref_energy),
                                                        finite_values(sim_energy)),
        "polymer_energy_distance" => scalar_distribution_energy(finite_values(ref_energy),
                                                                finite_values(sim_energy)),
        "reference_polymer_energy_mean" => Float32(mean(ref_energy)),
        "simulated_polymer_energy_mean" => Float32(mean(sim_energy)),
        "reference_pairwise_distance_mean" => Float32(mean(ref_pw)),
        "simulated_pairwise_distance_mean" => Float32(mean(sim_pw)),
    )

    return metrics, (; ref_energy, sim_energy, ref_pairwise=ref_pw, sim_pairwise=sim_pw)
end

function write_metrics_text(path::AbstractString, metrics)
    open(path, "w") do io
        for key in sort!(collect(keys(metrics)); by=string)
            println(io, key, " = ", metrics[key])
        end
    end
    return path
end

function plot_overlay(ref_values, sim_values, path::AbstractString; xlabel::AbstractString)
    fig = Figure(size=(900, 620), backgroundcolor=:white)
    ax = Axis(fig[1, 1], xlabel=xlabel, ylabel="density")
    ax.xgridvisible[] = false
    ax.ygridvisible[] = false
    hidespines!(ax, :t, :r)
    hist!(ax, finite_values(ref_values); bins=60, normalization=:pdf,
          color=(:gray30, 0.35), strokewidth=0, label="ground truth")
    hist!(ax, finite_values(sim_values); bins=60, normalization=:pdf,
          color=(:dodgerblue3, 0.45), strokewidth=0, label="learned-score sim")
    axislegend(ax, position=:rt)
    save(path, fig)
    return path
end

function run_brownian_rouse_trajectory(opts)
    checkpoint_path = project_path(String(opts["checkpoint"]))
    isfile(checkpoint_path) || error("checkpoint file does not exist: $checkpoint_path")
    checkpoint = deserialize(checkpoint_path)
    cfg = checkpoint["config"]
    seed = BoltzFlow.cfgint(cfg, "experiment.seed", 0)
    rng = Xoshiro(seed)

    polymer_cfg = if haskey(opts, "polymer_config")
        BoltzFlow.load_yaml_config(project_path(String(opts["polymer_config"])))
    else
        nothing
    end
    ckpt_phys = checkpoint_physics(cfg)
    exp_phys = experiment_physics(polymer_cfg)

    diffusion = Float64(get(opts, "diffusion",
                            exp_phys === nothing ? ckpt_phys.diffusion : exp_phys.diffusion))
    dt = Float64(get(opts, "dt", exp_phys === nothing ? ckpt_phys.dt : exp_phys.dt))
    steps = Int(opts["steps"])
    save_stride = Int(opts["save_stride"])
    log_interval_steps = Int(opts["log_interval_steps"])
    save_stride > 0 || error("--save-stride must be positive")
    steps > 0 || error("--steps must be positive")

    @info "loading model checkpoint" checkpoint_path
    flush(stderr)
    model_ctx = model_context_from_checkpoint(checkpoint)
    normalizer = BoltzFlow.cfgget(cfg, "data.normalizer",
                                  BoltzFlow.identity_data_normalizer())
    params = checkpoint["params"] |> model_ctx.device

    reference_data = nothing
    data_cfg = model_ctx.data_cfg
    try
        reference_data, data_cfg = load_reference_data(
            cfg, rng, Int(opts["max_data_samples"]))
    catch err
        if String(opts["init"]) == "data"
            @warn "could not load checkpoint training data; falling back to noise initialization" exception=(err, catch_backtrace())
            opts["init"] = "noise"
        else
            @warn "could not load checkpoint training data; metrics will be skipped" exception=(err, catch_backtrace())
        end
    end

    dim = data_cfg.dim
    n_atoms = data_cfg.n_atoms
    x0 = if String(opts["init"]) == "data" && reference_data !== nothing
        reference_data[:, :, rand(rng, axes(reference_data, 3))]
    else
        BoltzFlow.center_positions(randn(rng, Float32, dim, n_atoms))
    end

    score_fn = score_function(model_ctx, params, normalizer,
                              Float64(opts["diffusion_time"]))
    prob = learned_score_sde_problem(x0, dim, n_atoms, diffusion, dt, steps,
                                     score_fn;
                                     score_clip=Float64(opts["score_clip"]),
                                     dynamics=String(opts["dynamics"]),
                                     friction=Float64(opts["friction"]),
                                     mass=Float64(opts["mass"]),
                                     rng)
    saveat = collect(0.0:Float64(dt * save_stride):Float64(dt * steps))
    if saveat[end] < Float64(dt * steps)
        push!(saveat, Float64(dt * steps))
    end
    @info "solving learned-score Brownian Rouse trajectory" checkpoint_path family=model_ctx.family diffusion dt steps save_stride log_interval_steps frames=length(saveat)
    flush(stderr)
    solve_start = time()
    progress_cb = solve_progress_callback(steps, dt, log_interval_steps, solve_start)
    sol = progress_cb === nothing ?
          solve(prob, EM(); dt, saveat, adaptive=false) :
          solve(prob, EM(); dt, saveat, adaptive=false, callback=progress_cb)
    @info "finished solve" elapsed=format_duration(time() - solve_start) saved_frames=length(sol.u)
    flush(stderr)
    traj = solution_to_traj(sol, dim, n_atoms)
    times = Float64.(sol.t)

    output_dir = project_path(String(get(opts, "output_dir",
                                         latest_output_dir(checkpoint_path))))
    mkpath(output_dir)
    traj_path = joinpath(output_dir, "brownian_rouse_trajectory.jls")
    @info "writing trajectory payload" traj_path
    flush(stderr)
    serialize(traj_path, Dict{String,Any}(
        "checkpoint_path" => checkpoint_path,
        "checkpoint_epoch" => get(checkpoint, "epoch", nothing),
        "checkpoint_family" => String(model_ctx.family),
        "config" => cfg,
        "polymer_config" => polymer_cfg,
        "diffusion" => Float32(diffusion),
        "dt" => Float32(dt),
        "steps" => steps,
        "save_stride" => save_stride,
        "log_interval_steps" => log_interval_steps,
        "score_clip" => Float32(opts["score_clip"]),
        "diffusion_time" => Float32(opts["diffusion_time"]),
        "dynamics" => String(opts["dynamics"]),
        "friction" => Float32(opts["friction"]),
        "mass" => Float32(opts["mass"]),
        "times" => times,
        "traj" => traj,
        "created_at" => Dates.now(),
    ))

    draw_ring_bond = length(data_cfg.physics_params) >= 37 &&
                     data_cfg.physics_params[37] > 0.5f0
    frame_dir = save_chain_frames(traj, times, joinpath(output_dir, "frames");
                                  draw_ring_bond)
    gif_path = nothing
    if Bool(opts["gif"])
        gif_path = record_gif(traj, times, joinpath(output_dir, "brownian_rouse_trajectory.gif");
                              framerate=Int(opts["framerate"]), draw_ring_bond)
    end

    metrics_path = nothing
    if reference_data !== nothing
        @info "computing distribution metrics" output_dir
        flush(stderr)
        metrics, distributions = distribution_metrics(
            reference_data, traj, data_cfg, rng;
            metric_samples=Int(opts["metric_samples"]),
            metric_values=Int(opts["metric_values"]))
        merge!(metrics, Dict{String,Any}(
            "checkpoint_path" => checkpoint_path,
            "checkpoint_epoch" => get(checkpoint, "epoch", nothing),
            "checkpoint_family" => String(model_ctx.family),
            "diffusion" => Float32(diffusion),
            "dt" => Float32(dt),
            "steps" => steps,
            "save_stride" => save_stride,
            "dynamics" => String(opts["dynamics"]),
            "trajectory_path" => traj_path,
            "frame_dir" => frame_dir,
            "gif_path" => gif_path,
            "score_to_force" => "force = diffusion * grad_x log p_model(x)",
        ))
        serialize(joinpath(output_dir, "distribution_metrics.jls"), metrics)
        metrics_path = write_metrics_text(joinpath(output_dir, "distribution_metrics.txt"),
                                          metrics)
        serialize(joinpath(output_dir, "distributions.jls"), Dict{String,Any}(
            "reference_energy" => distributions.ref_energy,
            "simulated_energy" => distributions.sim_energy,
            "reference_pairwise_distance" => distributions.ref_pairwise,
            "simulated_pairwise_distance" => distributions.sim_pairwise,
        ))
        plot_overlay(distributions.ref_energy, distributions.sim_energy,
                     joinpath(output_dir, "polymer_energy_overlay.png");
                     xlabel="polymer energy")
        plot_overlay(distributions.ref_pairwise, distributions.sim_pairwise,
                     joinpath(output_dir, "pairwise_distance_overlay.png");
                     xlabel="pairwise distance")
        @info "finished distribution metrics" metrics_path
        flush(stderr)
    end

    println("trajectory: ", traj_path)
    println("frames:     ", frame_dir)
    gif_path === nothing || println("gif:        ", gif_path)
    metrics_path === nothing || println("metrics:    ", metrics_path)
    return (; traj_path, frame_dir, gif_path, metrics_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_brownian_rouse_trajectory(parse_args(ARGS))
end
