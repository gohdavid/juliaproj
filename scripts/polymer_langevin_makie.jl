#!/usr/bin/env julia

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "src", "BoltzFlow.jl"))

using .BoltzFlow
using CairoMakie
using Random
using Statistics
using Printf

function axis_limits(traj; pad=0.12f0)
    xmin, xmax = extrema(vec(traj[1, :, :]))
    ymin, ymax = extrema(vec(traj[2, :, :]))
    width = max(xmax - xmin, ymax - ymin)
    cx = (xmax + xmin) / 2
    cy = (ymax + ymin) / 2
    half = width * (0.5 + pad)
    return (cx - half, cx + half), (cy - half, cy + half)
end

function chain_points(traj, frame)
    return Point2f.(traj[1, :, frame], traj[2, :, frame])
end

function radius_of_gyration(traj)
    return vec(sqrt.(mean(sum(abs2, traj; dims=1); dims=2)))
end

function end_to_end_distance(traj)
    delta = traj[:, end, :] .- traj[:, 1, :]
    return vec(sqrt.(sum(abs2, delta; dims=1)))
end

function mean_bond_length(traj)
    bonds = traj[:, 2:end, :] .- traj[:, 1:end-1, :]
    return vec(mean(sqrt.(sum(abs2, bonds; dims=1)); dims=2))
end

function plot_snapshots(traj, times, path)
    frame_ids = round.(Int, range(1, size(traj, 3); length=6))
    xlims, ylims = axis_limits(traj)
    fig = Figure(size=(1200, 720), backgroundcolor=:white)
    for (panel, frame) in enumerate(frame_ids)
        row = panel <= 3 ? 1 : 2
        col = panel <= 3 ? panel : panel - 3
        ax = Axis(fig[row, col], aspect=DataAspect(),
                  title=@sprintf("t = %.1f", times[frame]),
                  xticksvisible=false, yticksvisible=false,
                  xticklabelsvisible=false, yticklabelsvisible=false)
        xlims!(ax, xlims...)
        ylims!(ax, ylims...)
        pts = chain_points(traj, frame)
        lines!(ax, pts, color=:gray25, linewidth=3)
        scatter!(ax, pts, color=1:length(pts), colormap=:viridis,
                 markersize=9, strokewidth=0)
    end
    save(path, fig)
    return path
end

function plot_trajectory(traj, times, path)
    xlims, ylims = axis_limits(traj)
    selected_beads = unique(round.(Int, range(1, size(traj, 2); length=5)))
    snapshot_frames = round.(Int, range(1, size(traj, 3); length=40))
    rg = radius_of_gyration(traj)
    ree = end_to_end_distance(traj)
    bonds = mean_bond_length(traj)

    fig = Figure(size=(1400, 840), backgroundcolor=:white)
    ax = Axis(fig[1:2, 1], aspect=DataAspect(),
              xlabel="x", ylabel="y", title="2D Rouse polymer Langevin trajectory")
    xlims!(ax, xlims...)
    ylims!(ax, ylims...)
    for frame in snapshot_frames
        lines!(ax, chain_points(traj, frame), color=(:gray35, 0.16), linewidth=1)
    end
    for bead in selected_beads
        path_pts = Point2f.(traj[1, bead, :], traj[2, bead, :])
        lines!(ax, path_pts, linewidth=2)
    end
    final_pts = chain_points(traj, size(traj, 3))
    lines!(ax, final_pts, color=:black, linewidth=4)
    scatter!(ax, final_pts, color=1:length(final_pts), colormap=:viridis,
             markersize=10, strokewidth=0)

    ax_rg = Axis(fig[1, 2], xlabel="time", ylabel="Rg", title="Radius of gyration")
    lines!(ax_rg, times, rg, color=:teal, linewidth=2)

    ax_ree = Axis(fig[2, 2], xlabel="time", ylabel="distance",
                  title="End-to-end and mean bond length")
    lines!(ax_ree, times, ree, color=:tomato, linewidth=2, label="end-to-end")
    lines!(ax_ree, times, bonds, color=:gray20, linewidth=2, label="mean bond")
    axislegend(ax_ree, position=:rt)
    save(path, fig)
    return path
end

function animate_trajectory(traj, times, path)
    xlims, ylims = axis_limits(traj)
    frame_ids = round.(Int, range(1, size(traj, 3); length=240))
    fig = Figure(size=(800, 800), backgroundcolor=:white)
    ax = Axis(fig[1, 1], aspect=DataAspect(), xlabel="x", ylabel="y",
              title="2D polymer Langevin dynamics")
    xlims!(ax, xlims...)
    ylims!(ax, ylims...)

    record(fig, path, frame_ids; framerate=30) do frame
        empty!(ax)
        pts = chain_points(traj, frame)
        lines!(ax, pts, color=:black, linewidth=4)
        scatter!(ax, pts, color=1:length(pts), colormap=:viridis,
                 markersize=11, strokewidth=0)
        text!(ax, xlims[1] + 0.04 * (xlims[2] - xlims[1]),
              ylims[2] - 0.06 * (ylims[2] - ylims[1]),
              text=@sprintf("t = %.1f", times[frame]), fontsize=24)
    end
    return path
end

function main()
    out_dir = joinpath(@__DIR__, "..", "outputs", "polymer_langevin_makie")
    mkpath(out_dir)

    rng = Xoshiro(7)
    cfg = NBodyDataConfig(
        kind=:polymer_langevin,
        dim=2,
        n_atoms=32,
        n_samples=0,
        burn_in=0,
        total_steps=80_000,
        dt=0.005,
        physics_params=Float32[1.0, 1.0],
    )
    save_stride = 100
    traj = run_polymer_langevin_sde_simulation(rng, cfg; save_stride)
    times = Float32.(0:save_stride:cfg.total_steps) .* cfg.dt

    snapshots_path = joinpath(out_dir, "polymer_snapshots.png")
    trajectory_path = joinpath(out_dir, "polymer_trajectory.png")
    animation_path = joinpath(out_dir, "polymer_trajectory.gif")

    plot_snapshots(traj, times, snapshots_path)
    plot_trajectory(traj, times, trajectory_path)
    animate_trajectory(traj, times, animation_path)

    rg = radius_of_gyration(traj)
    bonds = mean_bond_length(traj)
    println("Saved snapshots:  $snapshots_path")
    println("Saved trajectory: $trajectory_path")
    println("Saved animation:  $animation_path")
    @printf "Frames: %d, beads: %d, simulated time: %.1f\n" size(traj, 3) size(traj, 2) times[end]
    @printf "Mean Rg after first half: %.4f\n" mean(rg[(length(rg) ÷ 2):end])
    @printf "Mean bond length after first half: %.4f\n" mean(bonds[(length(bonds) ÷ 2):end])
end

main()
