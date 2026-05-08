#!/usr/bin/env julia

"""
Rouse-chain analytic Gaussian-potential diagnostic.

Model
-----
This script simulates a 2D free Rouse chain with `N` beads, adjacent harmonic
springs, uniform bead diffusion coefficient `D`, and center-of-mass-aligned
snapshots. In the same overdamped parameterization used by `src/nbody_data.jl`,

    dX_t = b(X_t) dt + sqrt(2D) dW_t

with independent Brownian noise for every bead coordinate and spring drift

    b_i(X) = k_over_xi * (X_{i+1} - 2X_i + X_{i-1})        interior beads
    b_1(X) = k_over_xi * (X_2 - X_1)
    b_N(X) = k_over_xi * (X_{N-1} - X_N).

Using the Einstein relation `D = kBT / xi`, the stationary density on the
centered subspace is a Gaussian chain

    p(X) ∝ exp(-U(X)),
    U(X) = (k_over_xi / (2D)) * sum_{i=1}^{N-1} ||X_{i+1} - X_i||^2.

The equilibrium score is analytic:

    score(X) = ∇_X log p(X) = b(X) / D = -∇U(X).

Equivalently, with the path-graph Laplacian `L`,

    U(X) = (k_over_xi / (2D)) * tr(X L Xᵀ),
    score(X) = -(k_over_xi / D) * X L.

The centered covariance is singular because the center-of-mass translation mode
is removed. The nonzero Rouse-mode eigenvalues are

    λ_p = 2 * (1 - cos(pi * p / N)),  p = 1, ..., N-1,

so the slowest relaxation time is

    τ_R = 1 / (k_over_xi * λ_1).

At equilibrium, `2U(X)` is chi-square distributed with `dim * (N - 1)` degrees
of freedom, so

    E[U] = dim * (N - 1) / 2.

This script simulates for many relaxation times, saves a chain image every
`0.1τ_R`, assembles those images into a video, and plots a histogram of sampled
`U(X)` values with the analytic equilibrium mean shown as a vertical line.
"""

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "src", "BoltzFlow.jl"))

using .BoltzFlow
using CairoMakie
using Random
using Statistics
using Printf
using StochasticDiffEq

function rouse_relaxation_time(n::Int, k_over_xi::Real)
    lambda1 = 2.0f0 * (1.0f0 - cos(Float32(pi) / Float32(n)))
    return 1.0f0 / (Float32(k_over_xi) * lambda1)
end

function analytic_potential(x, diffusion::Real, k_over_xi::Real)
    bonds = x[:, 2:end] .- x[:, 1:end-1]
    return Float32(k_over_xi) / (2.0f0 * Float32(diffusion)) * sum(abs2, bonds)
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

function simulate_long_rouse_chain(; rng=Xoshiro(11), n_beads=32, dim=2,
                                   diffusion=1.0f0, bond_length=1.0f0,
                                   n_relaxation_times=50.0f0,
                                   save_interval_tau=0.1f0,
                                   dt=0.01f0)
    k_over_xi = 3.0f0 * diffusion / bond_length^2
    tau_r = rouse_relaxation_time(n_beads, k_over_xi)
    total_time = n_relaxation_times * tau_r
    total_steps = ceil(Int, total_time / dt)

    cfg = NBodyDataConfig(
        kind=:polymer_langevin,
        dim=dim,
        n_atoms=n_beads,
        n_samples=0,
        burn_in=0,
        total_steps=total_steps,
        dt=dt,
        physics_params=Float32[diffusion, bond_length, k_over_xi],
    )

    prob = polymer_langevin_sde_problem(rng, cfg)
    saveat = Float32.(0.0f0:save_interval_tau:n_relaxation_times) .* tau_r
    sol = solve(prob, EM(); dt=dt, saveat, adaptive=false)

    traj = zeros(Float32, dim, n_beads, length(sol.u))
    for i in eachindex(sol.u)
        traj[:, :, i] = BoltzFlow.center_frame(reshape(sol.u[i], dim, n_beads))
    end
    return cfg, traj, saveat, tau_r, k_over_xi
end

function save_chain_frames(traj, times, tau_r, frame_dir)
    mkpath(frame_dir)
    xlims, ylims = axis_limits(traj)
    n_frames = size(traj, 3)
    n_beads = size(traj, 2)

    for frame in 1:n_frames
        fig = Figure(size=(700, 700), backgroundcolor=:white)
        ax = Axis(fig[1, 1], aspect=DataAspect(),
                  title=@sprintf("2D Rouse chain, t = %.1fτR", times[frame] / tau_r),
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

function plot_potential_histogram(potentials, expected_u, out_path)
    fig = Figure(size=(900, 620), backgroundcolor=:white)
    ax = Axis(fig[1, 1],
              xlabel="U(X)",
              ylabel="count",
              title="Rouse-chain Gaussian potential over long SDE trajectory")
    hist!(ax, potentials; bins=42, color=(:teal, 0.72), strokewidth=1,
          strokecolor=:white)
    vlines!(ax, [expected_u], color=:black, linestyle=:dash, linewidth=4,
            label=@sprintf("analytic E[U] = %.1f", expected_u))
    vlines!(ax, [mean(potentials)], color=:tomato, linewidth=4,
            label=@sprintf("sample mean = %.1f", mean(potentials)))
    axislegend(ax, position=:rt)
    save(out_path, fig)
    return out_path
end

function main()
    out_dir = joinpath(@__DIR__, "..", "outputs", "rouse_score_analytics")
    frame_dir = joinpath(out_dir, "frames")
    mkpath(out_dir)

    diffusion = 1.0f0
    bond_length = 1.0f0
    n_relaxation_times = 50.0f0
    save_interval_tau = 0.1f0
    cfg, traj, times, tau_r, k_over_xi = simulate_long_rouse_chain(
        diffusion=diffusion,
        bond_length=bond_length,
        n_beads=32,
        n_relaxation_times=n_relaxation_times,
        save_interval_tau=save_interval_tau,
        dt=0.01f0,
    )

    potentials = [analytic_potential(traj[:, :, i], diffusion, k_over_xi)
                  for i in axes(traj, 3)]
    expected_u = Float32(cfg.dim * (cfg.n_atoms - 1)) / 2.0f0

    frame_dir = save_chain_frames(traj, times, tau_r, frame_dir)
    video_path = joinpath(out_dir, "rouse_chain_0p1tau_frames.mp4")
    assembled_video = assemble_video(frame_dir, video_path)
    histogram_path = joinpath(out_dir, "rouse_potential_histogram.png")
    plot_potential_histogram(potentials, expected_u, histogram_path)

    println("Saved frame directory: $frame_dir")
    println("Saved histogram:       $histogram_path")
    if assembled_video !== nothing
        println("Saved video:           $assembled_video")
    end
    @printf "N beads: %d\n" cfg.n_atoms
    @printf "D: %.3f, bond_length: %.3f, k_over_xi: %.3f\n" diffusion bond_length k_over_xi
    @printf "tau_R: %.4f\n" tau_r
    @printf "simulated relaxation times: %.1f\n" n_relaxation_times
    @printf "saved interval: %.2f tau_R, frames: %d\n" save_interval_tau size(traj, 3)
    @printf "analytic E[U]: %.4f\n" expected_u
    @printf "sample mean U: %.4f\n" mean(potentials)
    @printf "sample std U: %.4f\n" std(potentials)
end

main()
