#!/usr/bin/env julia

"""
Analyze one raw Rouse-chain trajectory saved by `simulate_rouse_raw.jl`.

No center-of-mass correction is applied in this analysis. The saved coordinates
are raw free-chain positions, including center-of-mass diffusion. The Gaussian
Rouse potential

    U(X) = (k_over_xi / (2D)) * sum_{i=1}^{N-1} ||X_{i+1} - X_i||^2

depends only on bond vectors, so it is invariant to global translation and can
be evaluated directly on raw coordinates.

At equilibrium,

    2U(X) ~ χ²_ν,     ν = dim * (N - 1),
    E[U] = ν / 2,    Var(U) = ν / 2.

The clean equilibrium score is

    score(X) = ∇ log p(X) = -∇U(X) = drift(X) / D.

In Rouse normal modes, each score mode has variance `(k_over_xi / D) * λ_p`,
where `λ_p = 2 * (1 - cos(pi * p / N))`, `p = 1, ..., N - 1`.
Therefore `||score(X)||²` is a weighted chi-square random variable:

    ||score||² = sum_{p=1}^{N-1} sum_{a=1}^{dim}
                 ((k_over_xi / D) * λ_p) * χ²_{p,a}(1).

For the 2D chain used here, the two spatial copies for each Rouse mode combine
into one exponential random variable. The exact PDF is the hypoexponential
density

    f(s) = sum_p c_p / θ_p * exp(-s / θ_p),
    θ_p = 2 * (k_over_xi / D) * λ_p,
    c_p = prod_{q≠p} θ_p / (θ_p - θ_q).

For potential diagnostics, the trajectory is expected to be sampled at a long
interval, e.g. every `1τ_R` or `2τ_R`, where `τ_R` is the slowest Rouse
relaxation time. Samples spaced by `1τ_R` are not perfectly independent in a
strict statistical sense, but they are spaced at the natural slow-mode
relaxation scale. This script reports both the analytic independent-sample
standard error

    SE_independent = sqrt(Var(U) / M)

and the observed sample standard error

    SE_observed = std(U_samples) / sqrt(M).

A plot config chooses whether to make a video/GIF, a potential-density
histogram, a score-norm histogram, or a combination from a single experiment
output file. Video plots preserve center-of-mass drift by setting global axes
from the raw video coordinates.

Usage:

    julia --project=. scripts/rouse_score_analytics.jl configs/plots/rouse_video.yaml
    julia --project=. scripts/rouse_score_analytics.jl configs/plots/rouse_potential.yaml
    julia --project=. scripts/rouse_score_analytics.jl configs/plots/rouse_score.yaml
"""

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CairoMakie
using Distributions
using HDF5
using Serialization
using Statistics
using Printf

include(joinpath(@__DIR__, "..", "src", "config.jl"))
include(joinpath(@__DIR__, "..", "src", "BoltzFlow.jl"))
using .BoltzFlow

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))

function project_path(path::AbstractString)
    return isabspath(path) ? path : joinpath(PROJECT_ROOT, path)
end

function output_child_path(cfg, out_dir::AbstractString, path_key::AbstractString,
                           file_key::AbstractString, default_file::AbstractString)
    file = cfgget(cfg, file_key, nothing)
    file !== nothing && return joinpath(out_dir, String(file))

    legacy_path = cfgget(cfg, path_key, nothing)
    legacy_path !== nothing && return joinpath(out_dir, basename(String(legacy_path)))

    return joinpath(out_dir, default_file)
end

function seed_filename(filename::AbstractString, seed::Int)
    root, ext = splitext(filename)
    return string(root, "_seed", seed, ext)
end

function frame_hdf5_path(output_path::AbstractString)
    root, ext = splitext(output_path)
    return isempty(ext) ? string(output_path, ".h5") : string(root, ".h5")
end

function rouse_relaxation_time(n::Int, k_over_xi::Real)
    lambda1 = 2.0 * (1.0 - cos(pi / Float64(n)))
    return 1.0 / (Float64(k_over_xi) * lambda1)
end

function nonideal_param_vector(sim_cfg)
    nonideal = get(sim_cfg, "nonideal", nothing)
    nonideal === nothing && return Float32[]
    lj = get(nonideal, "lennard_jones", Dict())
    ev = get(nonideal, "excluded_volume", Dict())
    conf = get(nonideal, "confinement", Dict())
    return Float32[
        get(lj, "enabled", false) ? 1.0f0 : 0.0f0,
        Float32(get(lj, "epsilon", 0.0f0)),
        Float32(get(lj, "sigma", 1.0f0)),
        Float32(get(lj, "softening", 0.0f0)),
        Float32(get(lj, "cutoff", 0.0f0)),
        get(lj, "exclude_bonded", true) ? 1.0f0 : 0.0f0,
        get(lj, "shift", true) ? 1.0f0 : 0.0f0,
        get(ev, "enabled", false) ? 1.0f0 : 0.0f0,
        Float32(get(ev, "epsilon", 0.0f0)),
        Float32(get(ev, "sigma", 1.0f0)),
        Float32(get(ev, "softening", 0.0f0)),
        Float32(get(ev, "power", 12.0f0)),
        Float32(get(ev, "cutoff", 0.0f0)),
        get(ev, "exclude_bonded", true) ? 1.0f0 : 0.0f0,
        get(conf, "enabled", false) ? 1.0f0 : 0.0f0,
        Float32(get(conf, "strength", 0.0f0)),
        get(conf, "centered", true) ? 1.0f0 : 0.0f0,
    ]
end

function nonideal_enabled(sim_cfg)
    nonideal = get(sim_cfg, "nonideal", nothing)
    nonideal === nothing && return false
    return get(get(nonideal, "lennard_jones", Dict()), "enabled", false) ||
           get(get(nonideal, "excluded_volume", Dict()), "enabled", false) ||
           get(get(nonideal, "confinement", Dict()), "enabled", false)
end

function analytic_potential(x, diffusion::Real, k_over_xi::Real, sim_cfg)
    nonideal = BoltzFlow.polymer_nonideal_params(nonideal_param_vector(sim_cfg))
    return BoltzFlow.polymer_langevin_potential(x, diffusion, k_over_xi, nonideal)
end

function analytic_score(x, diffusion::Real, k_over_xi::Real, sim_cfg)
    score = zeros(Float32, size(x))
    nonideal = BoltzFlow.polymer_nonideal_params(nonideal_param_vector(sim_cfg))
    BoltzFlow.polymer_langevin_score!(score, x, diffusion, k_over_xi, nonideal)
    return score
end

function rouse_laplacian_eigenvalues(n_beads::Int)
    return [2.0f0 * (1.0f0 - cos(Float32(pi) * Float32(p) / Float32(n_beads)))
            for p in 1:(n_beads - 1)]
end

function score_norm2_mean_var(dim::Int, n_beads::Int, diffusion::Real,
                              k_over_xi::Real)
    lambdas = rouse_laplacian_eigenvalues(n_beads)
    weights = Float32(k_over_xi / diffusion) .* lambdas
    mean_val = Float32(dim) * sum(weights)
    var_val = 2.0f0 * Float32(dim) * sum(abs2, weights)
    return mean_val, var_val
end

function exact_score_norm2_pdf_2d(xs, n_beads::Int, diffusion::Real,
                                  k_over_xi::Real)
    lambdas = rouse_laplacian_eigenvalues(n_beads)
    theta = BigFloat.(2.0f0 .* Float32(k_over_xi / diffusion) .* lambdas)
    coeffs = BigFloat[]
    for i in eachindex(theta)
        c = big"1"
        for j in eachindex(theta)
            i == j && continue
            c *= theta[i] / (theta[i] - theta[j])
        end
        push!(coeffs, c)
    end
    return [Float64(sum((coeffs[i] / theta[i]) * exp(-BigFloat(x) / theta[i])
                        for i in eachindex(theta)))
            for x in xs]
end

function chain_points(x)
    return Point2f.(x[1, :], x[2, :])
end

function axis_limits(traj; pad=0.18f0)
    xmin, xmax = extrema(vec(traj[1, :, :]))
    ymin, ymax = extrema(vec(traj[2, :, :]))
    width = max(xmax - xmin, ymax - ymin)
    cx = (xmax + xmin) / 2
    cy = (ymax + ymin) / 2
    half = width * (0.5 + pad)
    return (cx - half, cx + half), (cy - half, cy + half)
end

function save_chain_frames(traj, times, tau_r, frame_dir)
    mkpath(frame_dir)
    xlims, ylims = axis_limits(traj)
    n_frames = size(traj, 3)
    n_beads = size(traj, 2)

    for frame in 1:n_frames
        fig = Figure(size=(700, 700), backgroundcolor=:white)
        ax = Axis(fig[1, 1], aspect=DataAspect(),
                  title=@sprintf("raw Rouse chain, t = %.2fτR", times[frame] / tau_r),
                  xlabel="x", ylabel="y")
        xlims!(ax, xlims...)
        ylims!(ax, ylims...)
        pts = chain_points(traj[:, :, frame])
        lines!(ax, pts, color=:gray20, linewidth=4)
        scatter!(ax, pts, color=1:n_beads, colormap=:viridis,
                 markersize=10, strokewidth=0)
        save(joinpath(frame_dir, @sprintf("frame_%05d.png", frame)), fig)
    end
    return frame_dir
end

function record_gif(traj, times, tau_r, gif_path; framerate=24)
    xlims, ylims = axis_limits(traj)
    n_beads = size(traj, 2)
    frame_ids = collect(axes(traj, 3))
    fig = Figure(size=(700, 700), backgroundcolor=:white)
    ax = Axis(fig[1, 1], aspect=DataAspect(), xlabel="x", ylabel="y",
              title="raw 2D Rouse chain")
    xlims!(ax, xlims...)
    ylims!(ax, ylims...)

    record(fig, gif_path, frame_ids; framerate) do frame
        empty!(ax)
        pts = chain_points(traj[:, :, frame])
        lines!(ax, pts, color=:gray20, linewidth=4)
        scatter!(ax, pts, color=1:n_beads, colormap=:viridis,
                 markersize=10, strokewidth=0)
        text!(ax, xlims[1] + 0.04 * (xlims[2] - xlims[1]),
              ylims[2] - 0.06 * (ylims[2] - ylims[1]),
              text=@sprintf("t = %.2fτR", times[frame] / tau_r), fontsize=24)
    end
    return gif_path
end

function assemble_video(frame_dir, video_path; framerate=24)
    ffmpeg = Sys.which("ffmpeg")
    if ffmpeg === nothing
        @warn "ffmpeg was not found; leaving PNG frame sequence unassembled." frame_dir
        return nothing
    end

    input_pattern = joinpath(frame_dir, "frame_%05d.png")
    run(`$ffmpeg -y -framerate $framerate -i $input_pattern -vf format=yuv420p $video_path`)
    return video_path
end

function plot_potential_histogram(potentials, expected_u, dof, se_independent,
                                  se_observed, out_path; ideal_reference::Bool=true)
    dist = Gamma(Float64(dof) / 2, 1.0)
    xmin = max(0.0, Float64(minimum(potentials)) - 0.15 * Float64(std(potentials)))
    xmax = Float64(maximum(potentials)) + 0.15 * Float64(std(potentials))
    xs = range(xmin, xmax; length=600)
    ys = pdf.(dist, xs)

    fig = Figure(size=(950, 650), backgroundcolor=:white)
    ax = Axis(fig[1, 1],
              xlabel="U(X)",
              ylabel="probability density",
              title="Rouse-chain Gaussian potential from raw long-interval samples")
    hist!(ax, potentials; bins=36, normalization=:pdf,
          color=(:teal, 0.55), strokewidth=1,
          strokecolor=:white)
    if ideal_reference
        lines!(ax, xs, ys, color=:black, linewidth=4,
               label=@sprintf("analytic Gamma(%.1f, 1) PDF", dof / 2))
        vlines!(ax, [expected_u], color=:black, linestyle=:dash, linewidth=4,
                label=@sprintf("analytic E[U] = %.2f", expected_u))
    end
    vlines!(ax, [mean(potentials)], color=:tomato, linewidth=4,
            label=@sprintf("sample mean = %.2f", mean(potentials)))
    axislegend(ax, position=:rt)
    if ideal_reference
        text!(ax, 0.03, 0.94, space=:relative, align=(:left, :top),
              text=@sprintf("analytic SE(mean) = %.3f\nobserved SE(mean) = %.3f",
                            se_independent, se_observed),
              fontsize=18)
    end
    save(out_path, fig)
    return out_path
end

function histogram_density(values, edges)
    counts = zeros(Float64, length(edges) - 1)
    for value in values
        if value == edges[end]
            counts[end] += 1
            continue
        end
        idx = searchsortedlast(edges, Float64(value))
        if 1 <= idx <= length(counts)
            counts[idx] += 1
        end
    end
    widths = diff(edges)
    density = counts ./ (sum(counts) .* widths)
    centers = (edges[1:end-1] .+ edges[2:end]) ./ 2
    return centers, density
end

function plot_score_norm2_histogram(score_norm2, xs, analytic_density,
                                    expected_norm2, out_path; bins::Int=60,
                                    ideal_reference::Bool=true)
    xmin = 0.0
    xmax = ideal_reference ? maximum(xs) : maximum(Float64.(score_norm2)) * 1.05
    sim_edges = collect(range(xmin, xmax; length=bins + 1))

    fig = Figure(size=(950, 650), backgroundcolor=:white)
    ax = Axis(fig[1, 1],
              xlabel="||score(X)||²",
              ylabel="probability density",
              title="Rouse-chain analytic score-norm distribution")
    hist!(ax, score_norm2; bins=sim_edges, normalization=:pdf,
          color=(:steelblue, 0.55), strokewidth=1,
          strokecolor=:white, label="simulation")
    if ideal_reference
        lines!(ax, xs, analytic_density, color=:black, linewidth=4,
               label="exact analytic PDF")
        vlines!(ax, [expected_norm2], color=:black, linestyle=:dash, linewidth=4,
                label=@sprintf("analytic E[||score||²] = %.2f", expected_norm2))
    end
    vlines!(ax, [mean(score_norm2)], color=:tomato, linewidth=4,
            label=@sprintf("sample mean = %.2f", mean(score_norm2)))
    axislegend(ax, position=:rt)
    save(out_path, fig)
    return out_path
end

function experiment_output_paths_and_dir(plot_cfg)
    explicit = cfgget(plot_cfg, "data.path", nothing)
    if explicit !== nothing
        data_path = project_path(String(explicit))
        return [data_path], dirname(data_path)
    end

    exp_cfg_path = cfgget(plot_cfg, "experiment.config", nothing)
    exp_cfg_path === nothing && error("Plot config needs either data.path or experiment.config")
    exp_cfg = load_yaml_config(project_path(String(exp_cfg_path)))
    output_dir = cfgget(exp_cfg, "output.dir", nothing)
    if output_dir !== nothing
        legacy_output = cfgget(exp_cfg, "output.path", nothing)
        default_file = legacy_output === nothing ? "rouse.jls" : basename(String(legacy_output))
        output_file = String(cfgget(exp_cfg, "output.file", default_file))
        data_dir = project_path(config_output_dir(exp_cfg; default_name="rouse_raw"))
        n_workers = cfgint(exp_cfg, "parallel.workers", 1)
        if n_workers > 1
            seed0 = cfgint(exp_cfg, "seed", 11)
            stride = cfgint(exp_cfg, "parallel.seed_stride", 1)
            paths = [joinpath(data_dir, seed_filename(output_file, seed0 + (worker_id - 1) * stride))
                     for worker_id in 1:n_workers]
            return paths, data_dir
        end
        return [joinpath(data_dir, output_file)], data_dir
    end
    data_path = project_path(String(cfgget(exp_cfg, "output.path", "outputs/rouse_raw/rouse.jls")))
    return [data_path], dirname(data_path)
end

function sim_config_from_experiment_config(exp_cfg)
    dim = cfgint(exp_cfg, "rouse.dim", 2)
    n_beads = cfgint(exp_cfg, "rouse.n_beads", 32)
    diffusion = cfgfloat32(exp_cfg, "rouse.diffusion", 1.0f0)
    bond_length = cfgfloat32(exp_cfg, "rouse.bond_length", 1.0f0)
    k_over_xi = cfgfloat32(exp_cfg, "rouse.k_over_xi",
                           3.0f0 * diffusion / bond_length^2)
    return Dict(
        "seed" => cfgint(exp_cfg, "seed", 11),
        "dim" => dim,
        "n_beads" => n_beads,
        "diffusion" => diffusion,
        "bond_length" => bond_length,
        "k_over_xi" => k_over_xi,
        "solver_algorithm" => String(cfgget(exp_cfg, "solver.algorithm", "EM")),
        "dt" => cfgfloat32(exp_cfg, "solver.dt", 0.01f0),
        "tau_r" => rouse_relaxation_time(n_beads, k_over_xi),
        "burn_in_tau" => cfgfloat32(exp_cfg, "save.burn_in_tau", 0.0f0),
        "until_tau" => cfgfloat32(exp_cfg, "save.until_tau", 5.0f0),
        "interval_tau" => cfgfloat32(exp_cfg, "save.interval_tau", 0.05f0),
        "nonideal" => Dict(
            "lennard_jones" => Dict(
                "enabled" => cfgbool(exp_cfg, "nonideal.lennard_jones.enabled", false),
                "epsilon" => cfgfloat32(exp_cfg, "nonideal.lennard_jones.epsilon", 0.0f0),
                "sigma" => cfgfloat32(exp_cfg, "nonideal.lennard_jones.sigma", 1.0f0),
                "softening" => cfgfloat32(exp_cfg, "nonideal.lennard_jones.softening", 0.0f0),
                "cutoff" => cfgfloat32(exp_cfg, "nonideal.lennard_jones.cutoff", 0.0f0),
                "exclude_bonded" => cfgbool(exp_cfg, "nonideal.lennard_jones.exclude_bonded", true),
                "shift" => cfgbool(exp_cfg, "nonideal.lennard_jones.shift", true),
            ),
            "excluded_volume" => Dict(
                "enabled" => cfgbool(exp_cfg, "nonideal.excluded_volume.enabled", false),
                "epsilon" => cfgfloat32(exp_cfg, "nonideal.excluded_volume.epsilon", 0.0f0),
                "sigma" => cfgfloat32(exp_cfg, "nonideal.excluded_volume.sigma", 1.0f0),
                "softening" => cfgfloat32(exp_cfg, "nonideal.excluded_volume.softening", 0.0f0),
                "power" => cfgfloat32(exp_cfg, "nonideal.excluded_volume.power", 12.0f0),
                "cutoff" => cfgfloat32(exp_cfg, "nonideal.excluded_volume.cutoff", 0.0f0),
                "exclude_bonded" => cfgbool(exp_cfg, "nonideal.excluded_volume.exclude_bonded", true),
            ),
            "confinement" => Dict(
                "enabled" => cfgbool(exp_cfg, "nonideal.confinement.enabled", false),
                "strength" => cfgfloat32(exp_cfg, "nonideal.confinement.strength", 0.0f0),
                "centered" => cfgbool(exp_cfg, "nonideal.confinement.centered", true),
            ),
        ),
    )
end

function hdf5_payload_path(jls_path::AbstractString)
    root, ext = splitext(jls_path)
    return ext == ".jls" ? string(root, ".h5") : frame_hdf5_path(jls_path)
end

function load_hdf5_payload(hdf5_path::AbstractString, sim_cfg)
    h5open(hdf5_path, "r") do h5
        written = haskey(h5, "written") ? Int(read(h5["written"])[1]) : size(h5["traj"], 3)
        written > 0 || error("No frames have been written in $hdf5_path")
        traj = h5["traj"][:, :, 1:written]
        times = h5["times"][1:written]
        times_tau = haskey(h5, "times_tau") ? h5["times_tau"][1:written] : times ./ sim_cfg["tau_r"]
        return Dict(
            "config_path" => hdf5_path,
            "config" => sim_cfg,
            "times" => times,
            "times_tau" => times_tau,
            "traj" => traj,
        )
    end
end

function load_payload(path::AbstractString, sim_cfg; prefer_hdf5::Bool=false)
    h5_path = hdf5_payload_path(path)
    if prefer_hdf5 && isfile(h5_path)
        return load_hdf5_payload(h5_path, sim_cfg)
    elseif isfile(path)
        return deserialize(path)
    elseif isfile(h5_path)
        return load_hdf5_payload(h5_path, sim_cfg)
    end
    error("Missing trajectory output: $path")
end

function combine_payloads(paths, sim_cfg; allow_partial::Bool=false, prefer_hdf5::Bool=false)
    payloads = Dict{String,Any}[]
    loaded_paths = String[]
    for path in paths
        try
            push!(payloads, load_payload(path, sim_cfg; prefer_hdf5))
            push!(loaded_paths, path)
        catch err
            if allow_partial
                @warn "Skipping unavailable trajectory output" path exception=(err, catch_backtrace())
            else
                rethrow()
            end
        end
    end
    isempty(payloads) && error("No trajectory outputs were available for analysis")
    ref_cfg = payloads[1]["config"]
    for payload in payloads[2:end]
        cfg = payload["config"]
        cfg["dim"] == ref_cfg["dim"] || error("Cannot combine runs with different dim")
        cfg["n_beads"] == ref_cfg["n_beads"] || error("Cannot combine runs with different n_beads")
        cfg["diffusion"] == ref_cfg["diffusion"] || error("Cannot combine runs with different diffusion")
        cfg["k_over_xi"] == ref_cfg["k_over_xi"] || error("Cannot combine runs with different k_over_xi")
    end
    traj = cat((payload["traj"] for payload in payloads)...; dims=3)
    times = vcat((payload["times"] for payload in payloads)...)
    times_tau = vcat((payload["times_tau"] for payload in payloads)...)
    cfg = copy(ref_cfg)
    cfg["combined_runs"] = length(payloads)
    cfg["combined_frames"] = size(traj, 3)
    return Dict(
        "config_path" => loaded_paths,
        "config" => cfg,
        "times" => times,
        "times_tau" => times_tau,
        "traj" => traj,
    )
end

function plot_output_dir(plot_cfg, data_dir::AbstractString)
    return joinpath(data_dir, "figures", config_hash(plot_cfg; exclude=("output",)))
end

function make_video_outputs(payload, out_dir, plot_cfg)
    traj = payload["traj"]
    times = payload["times"]
    sim_cfg = payload["config"]
    tau_r = sim_cfg["tau_r"]

    frame_dir = output_child_path(plot_cfg, out_dir, "output.frame_dir",
                                  "output.frame_dirname", "frames")
    gif_path = output_child_path(plot_cfg, out_dir, "output.gif_path",
                                 "output.gif_file", "rouse_raw.gif")
    mp4_path = output_child_path(plot_cfg, out_dir, "output.mp4_path",
                                 "output.mp4_file", "rouse_raw.mp4")
    framerate = cfgint(plot_cfg, "video.framerate", 24)

    save_chain_frames(traj, times, tau_r, frame_dir)
    record_gif(traj, times, tau_r, gif_path; framerate)
    assembled_video = assemble_video(frame_dir, mp4_path; framerate)

    println("Saved frame directory: $frame_dir")
    if assembled_video !== nothing
        println("Saved video:           $assembled_video")
    else
        println("Saved animation:       $gif_path")
    end
    @printf "video frames: %d through %.2f tau_R\n" size(traj, 3) (times[end] / tau_r)
end

function make_potential_output(payload, out_dir, plot_cfg)
    traj = payload["traj"]
    times = payload["times"]
    sim_cfg = payload["config"]
    diffusion = sim_cfg["diffusion"]
    k_over_xi = sim_cfg["k_over_xi"]
    dim = sim_cfg["dim"]
    n_beads = sim_cfg["n_beads"]
    tau_r = sim_cfg["tau_r"]

    potentials = [analytic_potential(traj[:, :, i], diffusion, k_over_xi, sim_cfg)
                  for i in axes(traj, 3)]
    dof = dim * (n_beads - 1)
    expected_u = Float32(dof) / 2.0f0
    analytic_var_u = Float32(dof) / 2.0f0
    n_samples = length(potentials)
    se_independent = sqrt(analytic_var_u / Float32(n_samples))
    se_observed = std(potentials) / sqrt(Float32(n_samples))

    ideal_reference = !nonideal_enabled(sim_cfg)
    histogram_path = output_child_path(plot_cfg, out_dir, "output.histogram_path",
                                       "output.histogram_file",
                                       "rouse_raw_potential_histogram.png")
    plot_potential_histogram(potentials, expected_u, dof, se_independent,
                             se_observed, histogram_path; ideal_reference)

    println("Saved histogram:       $histogram_path")
    @printf "N beads: %d\n" n_beads
    @printf "tau_R: %.4f\n" tau_r
    @printf "analysis samples: %d through %.2f tau_R\n" n_samples (times[end] / tau_r)
    ideal_reference && @printf "analytic E[U]: %.4f\n" expected_u
    @printf "sample mean U: %.4f\n" mean(potentials)
    @printf "sample std U: %.4f\n" std(potentials)
    ideal_reference && @printf "independent-sample SE(mean): %.4f\n" se_independent
    @printf "observed sample SE(mean): %.4f\n" se_observed
end

function make_score_output(payload, out_dir, plot_cfg)
    traj = payload["traj"]
    times = payload["times"]
    sim_cfg = payload["config"]
    diffusion = sim_cfg["diffusion"]
    k_over_xi = sim_cfg["k_over_xi"]
    dim = sim_cfg["dim"]
    n_beads = sim_cfg["n_beads"]
    tau_r = sim_cfg["tau_r"]

    score_norm2 = [sum(abs2, analytic_score(traj[:, :, i], diffusion, k_over_xi, sim_cfg))
                   for i in axes(traj, 3)]
    nonideal_enabled(sim_cfg) &&
        @warn "Score histogram uses total analytic non-ideal score, but ideal continuous Rouse PDF overlay is disabled."
    bins = cfgint(plot_cfg, "score.bins", 60)
    ideal_reference = !nonideal_enabled(sim_cfg)
    expected_norm2 = NaN32
    xs = Float64[]
    analytic_density = Float64[]
    if ideal_reference
        dim == 2 || error("Exact score-norm PDF is currently implemented for 2D Rouse chains.")
        expected_norm2, var_norm2 = score_norm2_mean_var(dim, n_beads, diffusion, k_over_xi)
        analytic_points = cfgint(plot_cfg, "score.analytic_points", 1200)
        xmax = max(maximum(Float64.(score_norm2)) * 1.05,
                   Float64(expected_norm2 + 4.0f0 * sqrt(var_norm2)))
        xs = collect(range(0.0, xmax; length=analytic_points))
        analytic_density = exact_score_norm2_pdf_2d(xs, n_beads, diffusion, k_over_xi)
    end

    score_path = output_child_path(plot_cfg, out_dir, "output.score_path",
                                   "output.score_file",
                                   "rouse_raw_score_norm2_histogram.png")
    plot_score_norm2_histogram(score_norm2, xs, analytic_density, expected_norm2,
                               score_path; bins=bins, ideal_reference)

    println("Saved score histogram: $score_path")
    @printf "N beads: %d\n" n_beads
    @printf "tau_R: %.4f\n" tau_r
    @printf "score samples: %d through %.2f tau_R\n" length(score_norm2) (times[end] / tau_r)
    ideal_reference && @printf "analytic E[||score||^2]: %.4f\n" expected_norm2
    @printf "sample mean ||score||^2: %.4f\n" mean(score_norm2)
    @printf "sample std ||score||^2: %.4f\n" std(score_norm2)
end

function main()
    length(ARGS) == 1 || error("Usage: julia --project=. scripts/rouse_score_analytics.jl <config.yaml>")
    plot_cfg_path = project_path(ARGS[1])
    plot_cfg = load_yaml_config(plot_cfg_path)

    data_paths, data_dir = experiment_output_paths_and_dir(plot_cfg)
    out_dir = plot_output_dir(plot_cfg, data_dir)
    mkpath(out_dir)

    exp_cfg_path = cfgget(plot_cfg, "experiment.config", nothing)
    exp_cfg = exp_cfg_path === nothing ? Dict{String,Any}() :
              load_yaml_config(project_path(String(exp_cfg_path)))
    sim_cfg = isempty(exp_cfg) ? Dict{String,Any}() : sim_config_from_experiment_config(exp_cfg)
    allow_partial = cfgbool(plot_cfg, "data.allow_partial", false)
    prefer_hdf5 = cfgbool(plot_cfg, "data.prefer_hdf5", false)
    payload = combine_payloads(data_paths, sim_cfg; allow_partial, prefer_hdf5)
    do_video = cfgbool(plot_cfg, "analysis.video", false)
    do_potential = cfgbool(plot_cfg, "analysis.potential", false)
    do_score = cfgbool(plot_cfg, "analysis.score", false)
    do_video || do_potential || do_score ||
        error("Plot config must set analysis.video, analysis.potential, or analysis.score to true")

    println("Loaded raw trajectory files: $(length(data_paths))")
    @printf "combined analysis frames: %d\n" size(payload["traj"], 3)
    do_video && make_video_outputs(payload, out_dir, plot_cfg)
    do_potential && make_potential_output(payload, out_dir, plot_cfg)
    do_score && make_score_output(payload, out_dir, plot_cfg)
end

main()
