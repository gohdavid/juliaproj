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
using Random
using Serialization
using Statistics
using Printf

include(joinpath(@__DIR__, "..", "src", "config.jl"))

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))

function project_path(path::AbstractString)
    return isabspath(path) ? path : joinpath(PROJECT_ROOT, path)
end

function analytic_potential(x, diffusion::Real, k_over_xi::Real)
    bonds = x[:, 2:end] .- x[:, 1:end-1]
    return Float32(k_over_xi) / (2.0f0 * Float32(diffusion)) * sum(abs2, bonds)
end

function analytic_score(x, diffusion::Real, k_over_xi::Real)
    score = zeros(Float32, size(x))
    @inbounds for i in 2:size(x, 2)
        r = x[:, i] .- x[:, i - 1]
        drift = Float32(k_over_xi) .* r
        score[:, i] .-= drift ./ Float32(diffusion)
        score[:, i - 1] .+= drift ./ Float32(diffusion)
    end
    return score
end

function rouse_laplacian_eigenvalues(n_beads::Int)
    return [2.0f0 * (1.0f0 - cos(Float32(pi) * Float32(p) / Float32(n_beads)))
            for p in 1:(n_beads - 1)]
end

function analytic_score_norm2_samples(rng, dim::Int, n_beads::Int,
                                      diffusion::Real, k_over_xi::Real,
                                      n_samples::Int)
    lambdas = rouse_laplacian_eigenvalues(n_beads)
    weights = Float32(k_over_xi / diffusion) .* lambdas
    samples = zeros(Float32, n_samples)
    for weight in weights, _ in 1:dim
        samples .+= weight .* Float32.(rand(rng, Chisq(1), n_samples))
    end
    return samples
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
                                  se_observed, out_path)
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
    lines!(ax, xs, ys, color=:black, linewidth=4,
           label=@sprintf("analytic Gamma(%.1f, 1) PDF", dof / 2))
    vlines!(ax, [expected_u], color=:black, linestyle=:dash, linewidth=4,
            label=@sprintf("analytic E[U] = %.2f", expected_u))
    vlines!(ax, [mean(potentials)], color=:tomato, linewidth=4,
            label=@sprintf("sample mean = %.2f", mean(potentials)))
    axislegend(ax, position=:rt)
    text!(ax, 0.03, 0.94, space=:relative, align=(:left, :top),
          text=@sprintf("analytic SE(mean) = %.3f\nobserved SE(mean) = %.3f",
                        se_independent, se_observed),
          fontsize=18)
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

function plot_score_norm2_histogram(score_norm2, analytic_norm2, expected_norm2,
                                    out_path; bins::Int=60, analytic_bins::Int=240)
    xmin = 0.0
    xmax = max(quantile(Float64.(analytic_norm2), 0.995),
               quantile(Float64.(score_norm2), 0.995))
    sim_edges = collect(range(xmin, xmax; length=bins + 1))
    analytic_edges = collect(range(xmin, xmax; length=analytic_bins + 1))
    centers, analytic_density = histogram_density(analytic_norm2, analytic_edges)

    fig = Figure(size=(950, 650), backgroundcolor=:white)
    ax = Axis(fig[1, 1],
              xlabel="||score(X)||²",
              ylabel="probability density",
              title="Rouse-chain analytic score-norm distribution")
    hist!(ax, score_norm2; bins=sim_edges, normalization=:pdf,
          color=(:steelblue, 0.55), strokewidth=1,
          strokecolor=:white, label="simulation")
    lines!(ax, centers, analytic_density, color=:black, linewidth=4,
           label="analytic weighted χ²")
    vlines!(ax, [expected_norm2], color=:black, linestyle=:dash, linewidth=4,
            label=@sprintf("analytic E[||score||²] = %.2f", expected_norm2))
    vlines!(ax, [mean(score_norm2)], color=:tomato, linewidth=4,
            label=@sprintf("sample mean = %.2f", mean(score_norm2)))
    axislegend(ax, position=:rt)
    save(out_path, fig)
    return out_path
end

function experiment_output_path(plot_cfg)
    explicit = cfgget(plot_cfg, "data.path", nothing)
    explicit !== nothing && return project_path(String(explicit))

    exp_cfg_path = cfgget(plot_cfg, "experiment.config", nothing)
    exp_cfg_path === nothing && error("Plot config needs either data.path or experiment.config")
    exp_cfg = load_yaml_config(project_path(String(exp_cfg_path)))
    return project_path(String(cfgget(exp_cfg, "output.path", "outputs/rouse_raw/rouse.jls")))
end

function make_video_outputs(payload, out_dir, plot_cfg)
    traj = payload["traj"]
    times = payload["times"]
    sim_cfg = payload["config"]
    tau_r = sim_cfg["tau_r"]

    frame_dir = project_path(String(cfgget(plot_cfg, "output.frame_dir",
                                           joinpath(out_dir, "frames"))))
    gif_path = project_path(String(cfgget(plot_cfg, "output.gif_path",
                                          joinpath(out_dir, "rouse_raw.gif"))))
    mp4_path = project_path(String(cfgget(plot_cfg, "output.mp4_path",
                                          joinpath(out_dir, "rouse_raw.mp4"))))
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

    potentials = [analytic_potential(traj[:, :, i], diffusion, k_over_xi)
                  for i in axes(traj, 3)]
    dof = dim * (n_beads - 1)
    expected_u = Float32(dof) / 2.0f0
    analytic_var_u = Float32(dof) / 2.0f0
    n_samples = length(potentials)
    se_independent = sqrt(analytic_var_u / Float32(n_samples))
    se_observed = std(potentials) / sqrt(Float32(n_samples))

    histogram_path = project_path(String(cfgget(plot_cfg, "output.histogram_path",
                                                joinpath(out_dir, "rouse_raw_potential_histogram.png"))))
    plot_potential_histogram(potentials, expected_u, dof, se_independent,
                             se_observed, histogram_path)

    println("Saved histogram:       $histogram_path")
    @printf "N beads: %d\n" n_beads
    @printf "tau_R: %.4f\n" tau_r
    @printf "analysis samples: %d through %.2f tau_R\n" n_samples (times[end] / tau_r)
    @printf "analytic E[U]: %.4f\n" expected_u
    @printf "sample mean U: %.4f\n" mean(potentials)
    @printf "sample std U: %.4f\n" std(potentials)
    @printf "independent-sample SE(mean): %.4f\n" se_independent
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

    score_norm2 = [sum(abs2, analytic_score(traj[:, :, i], diffusion, k_over_xi))
                   for i in axes(traj, 3)]
    lambdas = rouse_laplacian_eigenvalues(n_beads)
    expected_norm2 = Float32(dim) * Float32(k_over_xi / diffusion) * sum(lambdas)
    analytic_samples = cfgint(plot_cfg, "score.analytic_samples", 200000)
    bins = cfgint(plot_cfg, "score.bins", 60)
    analytic_bins = cfgint(plot_cfg, "score.analytic_bins", 240)
    analytic_norm2 = analytic_score_norm2_samples(
        Xoshiro(123), dim, n_beads, diffusion, k_over_xi, analytic_samples)

    score_path = project_path(String(cfgget(plot_cfg, "output.score_path",
                                            joinpath(out_dir, "rouse_raw_score_norm2_histogram.png"))))
    plot_score_norm2_histogram(score_norm2, analytic_norm2, expected_norm2,
                               score_path; bins=bins, analytic_bins=analytic_bins)

    println("Saved score histogram: $score_path")
    @printf "N beads: %d\n" n_beads
    @printf "tau_R: %.4f\n" tau_r
    @printf "score samples: %d through %.2f tau_R\n" length(score_norm2) (times[end] / tau_r)
    @printf "analytic E[||score||^2]: %.4f\n" expected_norm2
    @printf "sample mean ||score||^2: %.4f\n" mean(score_norm2)
    @printf "sample std ||score||^2: %.4f\n" std(score_norm2)
end

function main()
    length(ARGS) == 1 || error("Usage: julia --project=. scripts/rouse_score_analytics.jl <config.yaml>")
    plot_cfg_path = project_path(ARGS[1])
    plot_cfg = load_yaml_config(plot_cfg_path)

    out_dir = project_path(String(cfgget(plot_cfg, "output.dir", "outputs/rouse_raw")))
    mkpath(out_dir)

    data_path = experiment_output_path(plot_cfg)
    payload = deserialize(data_path)
    do_video = cfgbool(plot_cfg, "analysis.video", false)
    do_potential = cfgbool(plot_cfg, "analysis.potential", false)
    do_score = cfgbool(plot_cfg, "analysis.score", false)
    do_video || do_potential || do_score ||
        error("Plot config must set analysis.video, analysis.potential, or analysis.score to true")

    println("Loaded raw trajectory: $data_path")
    do_video && make_video_outputs(payload, out_dir, plot_cfg)
    do_potential && make_potential_output(payload, out_dir, plot_cfg)
    do_score && make_score_output(payload, out_dir, plot_cfg)
end

main()
